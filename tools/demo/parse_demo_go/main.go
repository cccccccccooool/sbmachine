// 6657 风格离线录像解说 AI 项目
// 项目功能：搭建一个"整段 CS2 录像 -> 分回合时间线 -> 人设 LLM 解说文本 -> GPT-SoVITS 语音"的离线生成流水线。
// 本文件功能：parse_demo_go: CS2 demo → JSON artefacts consumed by DemoQuery.
// Replaces tools/parse_demo.py (awpy). Uses markus-wa/demoinfocs-golang v5.
//
// 启动方式：被 tools/parse_demo.py 通过子进程调用（parse_demo_go --demo match.dem --output-dir output/demo）
// 输入数据流：CS2 .dem 文件。
// 输出数据流：output/demo/ 目录下的 JSON/jsonl 文件（demo_meta.json/rounds.json/kills.json 等）。
// 用法用途：解析 CS2 demo 文件，产出回合边界、击杀、道具、伤害、烟雾、燃烧弹、闪光、逐帧状态等 JSON 产物。
//
// NOTE: 使用 demoinfocs-golang v5.2.0，NewParserWithConfig 强制顺序解析（确定性输出）并开启 PacketEntities panic 容错。
//
// Usage:
//
//	parse_demo_go --demo match.dem --output-dir output/demo
//
// Outputs (all JSON / jsonl):
//
//	demo_meta.json     tick_rate, map_name, server_name
//	rounds.json        round boundaries + bomb events + alive counts
//	kills.json         kills + wallbang/smoke/noscope flags
//	grenades.json      throw+detonate per entity (legacy compat)
//	damages.json       per-hit damage events (PlayerHurt)
//	smokes.json        smoke start/end with world position
//	infernos.json      fire coverage hull + centroid
//	flashes.json       per-player flash events
//	ticks.jsonl        sampled player state (every ~0.5s)
//	roster.json        unique players (steamid, name, team)
//	callouts.json      parser-observed callout ID -> sampled-row count
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"

	dem "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/events"
	"github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/msg"
)

// ── output structs ────────────────────────────────────────────────────────────

type DemoMeta struct {
	TickRate        float64 `json:"tick_rate"`
	MapName         string  `json:"map_name"`
	ServerName      string  `json:"server_name"`
	DemoVersionName string  `json:"demo_version_name"`
}

type RoundRecord struct {
	RoundNo             int    `json:"round_no"`
	StartTick           int    `json:"start_tick"`
	FreezeEndTick       int    `json:"freeze_end_tick"`
	EndTick             int    `json:"end_tick"`
	BombPlantedTick     *int   `json:"bomb_planted_tick"`
	BombExplodedTick    *int   `json:"bomb_exploded_tick"`
	BombDefusedTick     *int   `json:"bomb_defused_tick"`
	BombBeginDefuseTick *int   `json:"bomb_begin_defuse_tick"`
	DefuserHasKit       *bool  `json:"defuser_has_kit"`
	Winner              string `json:"winner"`
	Reason              string `json:"reason"`
	BombSite            string `json:"bomb_site"`
	CTAliveEnd          int    `json:"ct_alive_end"`
	TAliveEnd           int    `json:"t_alive_end"`
}

type KillRecord struct {
	Tick            int     `json:"tick"`
	RoundNo         int     `json:"round_no"`
	Attacker        string  `json:"attacker"`
	AttackerSteamID string  `json:"attacker_steamid"`
	Victim          string  `json:"victim"`
	VictimSteamID   string  `json:"victim_steamid"`
	Assister        string  `json:"assister"`
	Weapon          string  `json:"weapon"`
	Headshot        bool    `json:"headshot"`
	ThroughSmoke    bool    `json:"through_smoke"`
	NoScope         bool    `json:"no_scope"`
	IsWallbang      bool    `json:"is_wallbang"`
	AttackerBlind   bool    `json:"attacker_blind"`
	AssistedFlash   bool    `json:"assisted_flash"`
	Distance        float64 `json:"distance"`
}

type GrenadeRecord struct {
	EntityID  string   `json:"entity_id"`
	RoundNo   int      `json:"round_no"`
	Thrower   string   `json:"thrower"`
	Type      string   `json:"type"`
	ThrowTick *int     `json:"throw_tick"`
	DetTick   *int     `json:"det_tick"`
	DestX     *float64 `json:"dest_x"`
	DestY     *float64 `json:"dest_y"`
	DestZ     *float64 `json:"dest_z"`
	ThrowPosX *float64 `json:"throw_pos_x"`
	ThrowPosY *float64 `json:"throw_pos_y"`
	ThrowPosZ *float64 `json:"throw_pos_z"`
}

type DamageRecord struct {
	Tick          int    `json:"tick"`
	RoundNo       int    `json:"round_no"`
	Attacker      string `json:"attacker"`
	AttackerSteam string `json:"attacker_steam"`
	Victim        string `json:"victim"`
	VictimSteam   string `json:"victim_steam"`
	HealthAfter   int    `json:"health_after"`
	ArmorAfter    int    `json:"armor_after"`
	HealthDmg     int    `json:"health_dmg"`
	ArmorDmg      int    `json:"armor_dmg"`
	HitGroup      string `json:"hit_group"`
	Weapon        string `json:"weapon"`
	ThroughSmoke  bool   `json:"through_smoke"`
}

type SmokeRecord struct {
	EntityID     int      `json:"entity_id"`
	RoundNo      int      `json:"round_no"`
	Thrower      string   `json:"thrower"`
	ThrowerSteam string   `json:"thrower_steam"`
	StartTick    int      `json:"start_tick"`
	EndTick      *int     `json:"end_tick"`
	PosX         float64  `json:"pos_x"`
	PosY         float64  `json:"pos_y"`
	PosZ         float64  `json:"pos_z"`
	DurationS    *float64 `json:"duration_s"`
}

type InfernoRecord struct {
	EntityID     int       `json:"entity_id"`
	RoundNo      int       `json:"round_no"`
	Thrower      string    `json:"thrower"`
	ThrowerSteam string    `json:"thrower_steam"`
	StartTick    int       `json:"start_tick"`
	EndTick      *int      `json:"end_tick"`
	HullX        []float64 `json:"hull_x"`
	HullY        []float64 `json:"hull_y"`
	CentroidX    float64   `json:"centroid_x"`
	CentroidY    float64   `json:"centroid_y"`
	AreaApprox   float64   `json:"area_approx"`
	DurationS    *float64  `json:"duration_s"`
}

type FlashRecord struct {
	Tick          int     `json:"tick"`
	RoundNo       int     `json:"round_no"`
	Victim        string  `json:"victim"`
	VictimSteam   string  `json:"victim_steam"`
	Attacker      string  `json:"attacker"`
	AttackerSteam string  `json:"attacker_steam"`
	DurationS     float64 `json:"duration_s"`
	IsTeammate    bool    `json:"is_teammate"`
}

type TickRecord struct {
	Tick         int     `json:"tick"`
	RoundNo      int     `json:"round_no"`
	SteamID      string  `json:"steamid"`
	Name         string  `json:"name"`
	Side         string  `json:"side"`
	X            float64 `json:"x"`
	Y            float64 `json:"y"`
	Z            float64 `json:"z"`
	HP           int     `json:"hp"`
	Armor        int     `json:"armor"`
	HasHelmet    bool    `json:"has_helmet"`
	HasKit       bool    `json:"has_kit"`
	ActiveWeapon string  `json:"active_weapon"`
	Ammo         int     `json:"ammo"`
	Money        int     `json:"money"`
	Callout      string  `json:"callout"`
	IsBlind      bool    `json:"is_blind"`
}

type RosterEntry struct {
	SteamID string `json:"steamid"`
	Name    string `json:"name"`
	Team    string `json:"team"`
}

// ── helpers ───────────────────────────────────────────────────────────────────

func ptrInt(v int) *int           { return &v }
func ptrFloat(v float64) *float64 { return &v }
func ptrBool(v bool) *bool        { return &v }

func roundEndReasonString(r events.RoundEndReason) string {
	switch r {
	case events.RoundEndReasonTargetBombed:
		return "TargetBombed"
	case events.RoundEndReasonBombDefused:
		return "BombDefused"
	case events.RoundEndReasonCTWin:
		return "CTWin"
	case events.RoundEndReasonTerroristsWin:
		return "TerroristsWin"
	case events.RoundEndReasonDraw:
		return "Draw"
	case events.RoundEndReasonHostagesRescued:
		return "HostagesRescued"
	case events.RoundEndReasonTargetSaved:
		return "TargetSaved"
	case events.RoundEndReasonTerroristsSurrender:
		return "TerroristsSurrender"
	case events.RoundEndReasonCTSurrender:
		return "CTSurrender"
	default:
		return fmt.Sprintf("Reason%d", r)
	}
}

func hitGroupName(hg events.HitGroup) string {
	switch hg {
	case events.HitGroupHead:
		return "Head"
	case events.HitGroupChest:
		return "Chest"
	case events.HitGroupStomach:
		return "Stomach"
	case events.HitGroupLeftArm:
		return "LeftArm"
	case events.HitGroupRightArm:
		return "RightArm"
	case events.HitGroupLeftLeg:
		return "LeftLeg"
	case events.HitGroupRightLeg:
		return "RightLeg"
	case events.HitGroupGear:
		return "Gear"
	default:
		return "Unknown"
	}
}

func sideStr(team common.Team) string {
	switch team {
	case common.TeamCounterTerrorists:
		return "CT"
	case common.TeamTerrorists:
		return "T"
	default:
		return ""
	}
}

func weaponName(w *common.Equipment) string {
	if w == nil {
		return ""
	}
	return w.Type.String()
}

func playerName(p *common.Player) string {
	if p == nil {
		return ""
	}
	return p.Name
}

func playerSteam(p *common.Player) string {
	if p == nil {
		return ""
	}
	return fmt.Sprintf("%d", p.SteamID64)
}

// convexHullArea computes the polygon area via shoelace formula.
func convexHullArea(xs, ys []float64) float64 {
	n := len(xs)
	if n < 3 {
		return 0
	}
	area := 0.0
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		area += xs[i] * ys[j]
		area -= xs[j] * ys[i]
	}
	return math.Abs(area) / 2.0
}

func writeJSON(path string, v any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		_ = f.Close()
		return err
	}
	return f.Close()
}

func validateEntitySnapshots(tickCount, rosterCount int) error {
	if tickCount == 0 || rosterCount == 0 {
		return fmt.Errorf(
			"DEM entity-state recovery failed: no participant snapshots (ticks=%d, roster=%d); PacketEntities may be incompatible with this demo",
			tickCount,
			rosterCount,
		)
	}
	return nil
}

// ── main ──────────────────────────────────────────────────────────────────────

func main() {
	demoPath := flag.String("demo", "", ".dem file path (required)")
	outDir := flag.String("output-dir", "output/demo", "output directory")
	flag.Parse()

	if *demoPath == "" {
		log.Fatal("--demo is required")
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		log.Fatalf("mkdir %s: %v", *outDir, err)
	}

	f, err := os.Open(*demoPath)
	if err != nil {
		log.Fatalf("open demo: %v", err)
	}
	defer f.Close()

	p := dem.NewParserWithConfig(f, dem.ParserConfig{
		MsgQueueBufferSize:             0,
		IgnorePacketEntitiesPanic:      true,
		DisableMimicSource1Events:      false,
		IgnoreErrBombsiteIndexNotFound: true,
	})
	defer p.Close()

	// ── state tracking ────────────────────────────────────────────────────────
	tickRate := 0.0
	mapName := ""
	serverName := ""

	currentRound := 0

	// 初始化为空切片而非 nil，空结果序列化为 [] 而非 null。
	rounds := []RoundRecord{}
	kills := []KillRecord{}
	grenades := []GrenadeRecord{}
	damages := []DamageRecord{}
	smokeList := []SmokeRecord{}
	infernos := []InfernoRecord{}
	flashes := []FlashRecord{}

	// per-round accumulators
	roundStart := map[int]int{}
	roundFreezeEnd := map[int]int{}
	roundEnd := map[int]int{}
	roundWinner := map[int]string{}
	roundReason := map[int]string{}
	roundBombSite := map[int]string{}
	roundBombPlanted := map[int]*int{}
	roundBombExploded := map[int]*int{}
	roundBombDefused := map[int]*int{}
	roundBombBeginDefuse := map[int]*int{}
	roundDefuserHasKit := map[int]*bool{}
	roundCTAlive := map[int]int{}
	roundTAlive := map[int]int{}

	// grenade entity tracking
	grenadeMap := map[int]*GrenadeRecord{}

	// smoke entity tracking
	smokeMap := map[int]*SmokeRecord{}

	// inferno entity tracking
	infernoMap := map[int]*InfernoRecord{}

	// roster
	rosterSeen := map[string]RosterEntry{}

	// ── event handlers ────────────────────────────────────────────────────────
	p.RegisterNetMessageHandler(func(e *msg.CSVCMsg_ServerInfo) {
		mapName = e.GetMapName()
		serverName = e.GetHostName()
	})

	p.RegisterEventHandler(func(e events.FrameDone) {
		tr := p.TickRate()
		if tr > 0 {
			tickRate = tr
		}
	})

	p.RegisterEventHandler(func(e events.RoundStart) {
		gs := p.GameState()
		currentRound = gs.TotalRoundsPlayed() + 1
		tick := p.GameState().IngameTick()
		roundStart[currentRound] = tick
		roundFreezeEnd[currentRound] = tick // 会被 RoundFreezetimeEnd 更新
	})

	p.RegisterEventHandler(func(e events.RoundFreezetimeEnd) {
		tick := p.GameState().IngameTick()
		roundFreezeEnd[currentRound] = tick
	})

	p.RegisterEventHandler(func(e events.RoundEnd) {
		tick := p.GameState().IngameTick()
		roundEnd[currentRound] = tick
		roundWinner[currentRound] = sideStr(e.Winner)
		roundReason[currentRound] = roundEndReasonString(e.Reason)
		// count alive players
		ct, t := 0, 0
		for _, pl := range p.GameState().Participants().Playing() {
			if pl.IsAlive() {
				switch pl.Team {
				case common.TeamCounterTerrorists:
					ct++
				case common.TeamTerrorists:
					t++
				}
			}
		}
		roundCTAlive[currentRound] = ct
		roundTAlive[currentRound] = t
	})

	p.RegisterEventHandler(func(e events.Kill) {
		tick := p.GameState().IngameTick()
		dist := 0.0
		if e.Killer != nil && e.Victim != nil {
			kp := e.Killer.Position()
			vp := e.Victim.Position()
			dx := kp.X - vp.X
			dy := kp.Y - vp.Y
			dz := kp.Z - vp.Z
			dist = math.Sqrt(dx*dx + dy*dy + dz*dz)
		}
		kills = append(kills, KillRecord{
			Tick:            tick,
			RoundNo:         currentRound,
			Attacker:        playerName(e.Killer),
			AttackerSteamID: playerSteam(e.Killer),
			Victim:          playerName(e.Victim),
			VictimSteamID:   playerSteam(e.Victim),
			Assister:        playerName(e.Assister),
			Weapon:          weaponName(e.Weapon),
			Headshot:        e.IsHeadshot,
			ThroughSmoke:    e.ThroughSmoke,
			NoScope:         e.NoScope,
			IsWallbang:      e.PenetratedObjects > 0,
			AttackerBlind:   e.AttackerBlind,
			AssistedFlash:   e.AssistedFlash,
			Distance:        math.Round(dist*10) / 10,
		})
	})

	p.RegisterEventHandler(func(e events.PlayerHurt) {
		tick := p.GameState().IngameTick()
		victim := e.Player
		dmg := DamageRecord{
			Tick:          tick,
			RoundNo:       currentRound,
			Victim:        playerName(victim),
			VictimSteam:   playerSteam(victim),
			Attacker:      playerName(e.Attacker),
			AttackerSteam: playerSteam(e.Attacker),
			HealthDmg:     e.HealthDamage,
			ArmorDmg:      e.ArmorDamage,
			HitGroup:      hitGroupName(e.HitGroup),
			Weapon:        weaponName(e.Weapon),
		}
		if victim != nil {
			dmg.HealthAfter = victim.Health()
			dmg.ArmorAfter = victim.Armor()
		}
		damages = append(damages, dmg)
	})

	p.RegisterEventHandler(func(e events.BombPlanted) {
		tick := p.GameState().IngameTick()
		roundBombPlanted[currentRound] = ptrInt(tick)
		if e.BombEvent.Site == 'A' {
			roundBombSite[currentRound] = "A"
		} else {
			roundBombSite[currentRound] = "B"
		}
	})

	p.RegisterEventHandler(func(e events.BombExplode) {
		tick := p.GameState().IngameTick()
		roundBombExploded[currentRound] = ptrInt(tick)
	})

	p.RegisterEventHandler(func(e events.BombDefused) {
		tick := p.GameState().IngameTick()
		roundBombDefused[currentRound] = ptrInt(tick)
	})

	p.RegisterEventHandler(func(e events.BombDefuseStart) {
		tick := p.GameState().IngameTick()
		roundBombBeginDefuse[currentRound] = ptrInt(tick)
		roundDefuserHasKit[currentRound] = ptrBool(e.HasKit)
	})

	p.RegisterEventHandler(func(e events.GrenadeProjectileThrow) {
		proj := e.Projectile
		if proj == nil {
			return
		}
		tick := p.GameState().IngameTick()
		eid := proj.UniqueID()
		pos := proj.Position()
		throwerName := playerName(proj.Thrower)
		grenadeMap[int(eid)] = &GrenadeRecord{
			EntityID:  fmt.Sprintf("%d", eid),
			RoundNo:   currentRound,
			Thrower:   throwerName,
			Type:      proj.WeaponInstance.Type.String(),
			ThrowTick: ptrInt(tick),
			ThrowPosX: ptrFloat(pos.X),
			ThrowPosY: ptrFloat(pos.Y),
			ThrowPosZ: ptrFloat(pos.Z),
		}
	})

	p.RegisterEventHandler(func(e events.GrenadeProjectileDestroy) {
		proj := e.Projectile
		if proj == nil {
			return
		}
		tick := p.GameState().IngameTick()
		eid := int(proj.UniqueID())
		pos := proj.Position()
		if rec, ok := grenadeMap[eid]; ok {
			rec.DetTick = ptrInt(tick)
			rec.DestX = ptrFloat(pos.X)
			rec.DestY = ptrFloat(pos.Y)
			rec.DestZ = ptrFloat(pos.Z)
		}
	})

	p.RegisterEventHandler(func(e events.SmokeStart) {
		tick := p.GameState().IngameTick()
		eid := e.GrenadeEntityID
		pos := e.Position
		smokeMap[eid] = &SmokeRecord{
			EntityID:     eid,
			RoundNo:      currentRound,
			Thrower:      playerName(e.Thrower),
			ThrowerSteam: playerSteam(e.Thrower),
			StartTick:    tick,
			PosX:         pos.X,
			PosY:         pos.Y,
			PosZ:         pos.Z,
		}
	})

	p.RegisterEventHandler(func(e events.SmokeExpired) {
		tick := p.GameState().IngameTick()
		eid := e.GrenadeEntityID
		if rec, ok := smokeMap[eid]; ok {
			rec.EndTick = ptrInt(tick)
			if tickRate > 0 {
				dur := float64(tick-rec.StartTick) / tickRate
				rec.DurationS = ptrFloat(math.Round(dur*100) / 100)
			}
			smokeList = append(smokeList, *rec)
			delete(smokeMap, eid)
		}
	})

	p.RegisterEventHandler(func(e events.InfernoStart) {
		inf := e.Inferno
		if inf == nil {
			return
		}
		tick := p.GameState().IngameTick()
		eid := int(inf.UniqueID())
		// Fires() 含已熄灭火点（past + present），只圈仍在燃烧的，否则火焰范围虚大。
		hull := inf.Fires().Active().ConvexHull2D()
		xs := make([]float64, len(hull))
		ys := make([]float64, len(hull))
		cx, cy := 0.0, 0.0
		for i, v := range hull {
			xs[i] = v.X
			ys[i] = v.Y
			cx += v.X
			cy += v.Y
		}
		if len(hull) > 0 {
			cx /= float64(len(hull))
			cy /= float64(len(hull))
		}
		infernoMap[eid] = &InfernoRecord{
			EntityID:     eid,
			RoundNo:      currentRound,
			Thrower:      playerName(inf.Thrower()),
			ThrowerSteam: playerSteam(inf.Thrower()),
			StartTick:    tick,
			HullX:        xs,
			HullY:        ys,
			CentroidX:    math.Round(cx*10) / 10,
			CentroidY:    math.Round(cy*10) / 10,
			AreaApprox:   math.Round(convexHullArea(xs, ys)*10) / 10,
		}
	})

	p.RegisterEventHandler(func(e events.InfernoExpired) {
		inf := e.Inferno
		if inf == nil {
			return
		}
		tick := p.GameState().IngameTick()
		eid := int(inf.UniqueID())
		if rec, ok := infernoMap[eid]; ok {
			rec.EndTick = ptrInt(tick)
			if tickRate > 0 {
				dur := float64(tick-rec.StartTick) / tickRate
				rec.DurationS = ptrFloat(math.Round(dur*100) / 100)
			}
			infernos = append(infernos, *rec)
			delete(infernoMap, eid)
		}
	})

	p.RegisterEventHandler(func(e events.PlayerFlashed) {
		tick := p.GameState().IngameTick()
		victim := e.Player
		atk := e.Attacker
		sameTeam := false
		if victim != nil && atk != nil {
			sameTeam = victim.Team == atk.Team
		}
		dur := e.FlashDuration().Seconds()
		flashes = append(flashes, FlashRecord{
			Tick:          tick,
			RoundNo:       currentRound,
			Victim:        playerName(victim),
			VictimSteam:   playerSteam(victim),
			Attacker:      playerName(atk),
			AttackerSteam: playerSteam(atk),
			DurationS:     math.Round(dur*100) / 100,
			IsTeammate:    sameTeam,
		})
	})

	// ── tick sampling ─────────────────────────────────────────────────────────
	ticksFile, err := os.Create(filepath.Join(*outDir, "ticks.jsonl"))
	if err != nil {
		log.Fatalf("create ticks.jsonl: %v", err)
	}
	tickEncoder := json.NewEncoder(ticksFile)
	tickEncoder.SetEscapeHTML(false)
	var tickWriteErr error

	sampleEvery := 32 // ~0.5s at 64 tick; adjusted after header parse
	tickCount := 0
	// This is an observed index, not a hard-coded map catalogue. The exact IDs
	// come from Player.LastPlaceName() / m_szLastPlaceName in this demo.
	calloutCounts := map[string]int{}

	p.RegisterEventHandler(func(e events.FrameDone) {
		if tickWriteErr != nil {
			return
		}
		tr := p.TickRate()
		if tr > 0 {
			sampleEvery = int(math.Round(tr / 2))
			if sampleEvery < 1 {
				sampleEvery = 1
			}
		}
		tick := p.GameState().IngameTick()
		if tick%sampleEvery != 0 {
			return
		}
		for _, pl := range p.GameState().Participants().Playing() {
			if pl == nil {
				continue
			}
			pos := pl.Position()
			wpn := ""
			ammo := -1 // -1 表示无持枪（下游 kill_semantics 按武器名过滤，不会误读）
			if aw := pl.ActiveWeapon(); aw != nil {
				wpn = aw.Type.String()
				ammo = aw.AmmoInMagazine()
			}
			rec := TickRecord{
				Tick:         tick,
				RoundNo:      currentRound,
				SteamID:      playerSteam(pl),
				Name:         pl.Name,
				Side:         sideStr(pl.Team),
				X:            math.Round(pos.X*10) / 10,
				Y:            math.Round(pos.Y*10) / 10,
				Z:            math.Round(pos.Z*10) / 10,
				HP:           pl.Health(),
				Armor:        pl.Armor(),
				HasHelmet:    pl.HasHelmet(),
				HasKit:       pl.HasDefuseKit(),
				ActiveWeapon: wpn,
				Ammo:         ammo,
				Money:        pl.Money(),
				Callout:      pl.LastPlaceName(),
				IsBlind:      pl.IsBlinded(),
			}
			if err := tickEncoder.Encode(rec); err != nil {
				tickWriteErr = err
				return
			}
			tickCount++
			if rec.Callout != "" {
				calloutCounts[rec.Callout]++
			}
			// update roster
			sid := playerSteam(pl)
			if _, seen := rosterSeen[sid]; !seen {
				rosterSeen[sid] = RosterEntry{
					SteamID: sid,
					Name:    pl.Name,
					Team:    sideStr(pl.Team),
				}
			}
		}
	})

	// ── parse ─────────────────────────────────────────────────────────────────
	fmt.Printf("Parsing %s ...\n", *demoPath)
	var parseErr error
	func() {
		defer func() {
			if r := recover(); r != nil {
				parseErr = fmt.Errorf("parse panic: %v", r)
			}
		}()
		parseErr = p.ParseToEnd()
		if parseErr == io.EOF {
			parseErr = nil
		}
	}()
	if parseErr != nil {
		log.Fatalf("parse demo: %v", parseErr)
	}
	if tickWriteErr != nil {
		log.Fatalf("encode ticks.jsonl: %v", tickWriteErr)
	}
	if err := ticksFile.Close(); err != nil {
		log.Fatalf("close ticks.jsonl: %v", err)
	}
	if err := validateEntitySnapshots(tickCount, len(rosterSeen)); err != nil {
		log.Fatal(err)
	}

	if tickRate == 0 {
		tickRate = 64
	}

	// ── assemble rounds ───────────────────────────────────────────────────────
	allRounds := map[int]bool{}
	for rn := range roundStart {
		allRounds[rn] = true
	}
	for rn := range roundEnd {
		allRounds[rn] = true
	}
	for rno := 1; rno <= len(allRounds); rno++ {
		st := roundStart[rno]
		fe := roundFreezeEnd[rno]
		if fe == 0 {
			fe = st
		}
		en := roundEnd[rno]
		rec := RoundRecord{
			RoundNo:             rno,
			StartTick:           st,
			FreezeEndTick:       fe,
			EndTick:             en,
			BombPlantedTick:     roundBombPlanted[rno],
			BombExplodedTick:    roundBombExploded[rno],
			BombDefusedTick:     roundBombDefused[rno],
			BombBeginDefuseTick: roundBombBeginDefuse[rno],
			DefuserHasKit:       roundDefuserHasKit[rno],
			Winner:              roundWinner[rno],
			Reason:              roundReason[rno],
			BombSite:            roundBombSite[rno],
			CTAliveEnd:          roundCTAlive[rno],
			TAliveEnd:           roundTAlive[rno],
		}
		rounds = append(rounds, rec)
	}

	// carry remaining smokes/infernos (ended mid-match without event)
	for _, rec := range smokeMap {
		smokeList = append(smokeList, *rec)
	}
	for _, rec := range infernoMap {
		infernos = append(infernos, *rec)
	}

	// flatten grenades map
	for _, rec := range grenadeMap {
		grenades = append(grenades, *rec)
	}

	// ── write outputs ─────────────────────────────────────────────────────────
	meta := DemoMeta{
		TickRate:        tickRate,
		MapName:         mapName,
		ServerName:      serverName,
		DemoVersionName: "CS2",
	}
	if err := writeJSON(filepath.Join(*outDir, "demo_meta.json"), meta); err != nil {
		log.Fatalf("write demo_meta: %v", err)
	}
	fmt.Printf("  meta: tick_rate=%.0f  map=%s\n", tickRate, mapName)

	if err := writeJSON(filepath.Join(*outDir, "rounds.json"), rounds); err != nil {
		log.Fatalf("write rounds: %v", err)
	}
	fmt.Printf("  rounds: %d\n", len(rounds))

	roster := make([]RosterEntry, 0, len(rosterSeen))
	for _, r := range rosterSeen {
		roster = append(roster, r)
	}
	if err := writeJSON(filepath.Join(*outDir, "roster.json"), roster); err != nil {
		log.Fatalf("write roster: %v", err)
	}
	fmt.Printf("  roster: %d players\n", len(roster))

	if err := writeJSON(filepath.Join(*outDir, "kills.json"), kills); err != nil {
		log.Fatalf("write kills: %v", err)
	}
	fmt.Printf("  kills: %d\n", len(kills))

	if err := writeJSON(filepath.Join(*outDir, "grenades.json"), grenades); err != nil {
		log.Fatalf("write grenades: %v", err)
	}
	fmt.Printf("  grenades: %d\n", len(grenades))

	if err := writeJSON(filepath.Join(*outDir, "damages.json"), damages); err != nil {
		log.Fatalf("write damages: %v", err)
	}
	fmt.Printf("  damages: %d\n", len(damages))

	if err := writeJSON(filepath.Join(*outDir, "smokes.json"), smokeList); err != nil {
		log.Fatalf("write smokes: %v", err)
	}
	fmt.Printf("  smokes: %d\n", len(smokeList))

	if err := writeJSON(filepath.Join(*outDir, "infernos.json"), infernos); err != nil {
		log.Fatalf("write infernos: %v", err)
	}
	fmt.Printf("  infernos: %d\n", len(infernos))

	if err := writeJSON(filepath.Join(*outDir, "flashes.json"), flashes); err != nil {
		log.Fatalf("write flashes: %v", err)
	}
	fmt.Printf("  flashes: %d\n", len(flashes))

	// A compact exact index for manual map initialization. Full coordinates
	// remain in ticks.jsonl for later calibration.
	if err := writeJSON(filepath.Join(*outDir, "callouts.json"), calloutCounts); err != nil {
		log.Fatalf("write callouts: %v", err)
	}
	fmt.Printf("  callouts: %d observed IDs\n", len(calloutCounts))

	fmt.Printf("  ticks.jsonl: %d rows\n", tickCount)
	fmt.Printf("\nDone → %s\n", *outDir)
}
