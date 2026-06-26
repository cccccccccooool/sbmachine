"""全场滚动记忆管理器。

启动方式：被 sbmachine/phase3b_style.py 的 run_phase3b() 导入调用。
输入数据流：demo rounds.json（含 winner/kills）和 phase3a 产出的 hype 值。
输出数据流：render() 返回文本摘要注入 prompt；emotion_snapshot() 返回情绪积累值供 phase3b 读取。
用法用途：逐局滚动更新全场记忆（比分/气势/击杀榜/转折点/情绪积累），注入每局解说 prompt。

# 新增：情绪积累状态（沮丧/愤怒），权重全部从 Prompt/json/hype_rules.json 读取。
# 沮丧/愤怒仅由全场积累驱动，不来自单局事件，每次增量极小。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_HALF_SIZE = 12
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


from sbmachine.common import load_hype_rules


def _half_label(round_no: int) -> str:
    """根据回合号返回半场标签（上半场/下半场/加时赛）。"""
    if round_no <= _HALF_SIZE:
        return f"上半场第{round_no}局"
    if round_no <= _HALF_SIZE * 2:
        return f"下半场第{round_no - _HALF_SIZE}局"
    return f"加时赛第{round_no - _HALF_SIZE * 2}局"


@dataclass
class MatchMemory:
    map_name:         str = "Unknown"
    total_rounds_est: int = 0
    _score_ct:        int = field(default=0, repr=False)
    _score_t:         int = field(default=0, repr=False)
    _winner_seq:      list[str]  = field(default_factory=list, repr=False)
    _turning_points:  list[str]  = field(default_factory=list, repr=False)
    _top_kills:       dict[str, int] = field(default_factory=dict, repr=False)
    # ── 情绪积累（全5种，全场微量积累） ──
    _emotion_acc: dict[str, float] = field(
        default_factory=lambda: {"沮丧": 0.0, "愤怒": 0.0},
        repr=False,
    )
    _last_round_hype: float = field(default=0.0, repr=False)

    @classmethod
    def init(cls, map_name: str = "Unknown", total_rounds_est: int = 0) -> "MatchMemory":
        return cls(map_name=map_name, total_rounds_est=total_rounds_est)

    # ── render ──

    def render(self) -> str:
        """返回注入 prompt 的只读文本摘要（含情绪积累状态）。"""
        if not self._winner_seq:
            return f"比赛地图：{self.map_name}，第1局，尚无历史数据。"

        round_no = len(self._winner_seq) + 1
        score_line = f"CT {self._score_ct} : {self._score_t} T"
        half = _half_label(round_no)

        momentum = ""
        if len(self._winner_seq) >= 2:
            streak = 0
            for w in reversed(self._winner_seq):
                if w == self._winner_seq[-1]:
                    streak += 1
                else:
                    break
            if streak >= 2:
                momentum = f"{self._winner_seq[-1]}方连下{streak}局，势头在{self._winner_seq[-1]}"

        is_mp = self._is_match_point(round_no)
        mp_str = "（赛点局）" if is_mp else ""

        top_players = sorted(self._top_kills.items(), key=lambda x: -x[1])[:3]
        form_str = "、".join(f"{n} {k}杀" for n, k in top_players) if top_players else ""
        tp_str = "；".join(self._turning_points[-3:]) if self._turning_points else ""

        lines = [f"地图：{self.map_name}  {half}{mp_str}  比分：{score_line}"]
        if momentum:
            lines.append(f"气势：{momentum}")
        if form_str:
            lines.append(f"本场表现：{form_str}")
        if tp_str:
            lines.append(f"关键转折：{tp_str}")

        # 情绪积累状态（仅在有意义时输出，避免污染正常局）
        rules = load_hype_rules()
        em_thresholds = {k: v["threshold"] for k, v in rules["emotions"].items()}
        sad_val = round(self._emotion_acc.get("沮丧", 0.0), 3)
        ang_val = round(self._emotion_acc.get("愤怒", 0.0), 3)
        if sad_val >= em_thresholds["沮丧"] or ang_val >= em_thresholds["愤怒"]:
            lines.append(f"主播情绪积累：沮丧={sad_val:.2f}  愤怒={ang_val:.2f}（满1.0；仅供风格模型参考）")

        return "\n".join(lines)

    def emotion_snapshot(self) -> dict[str, float]:
        return dict(self._emotion_acc)

    # ── update ──

    def update(self, round_record, demo_rounds: list[dict] | None = None, round_hype: float = 0.0) -> None:
        rn = int(getattr(round_record, "round_no", 0))
        demo_round = self._find_demo_round(rn, demo_rounds)
        winner = str(demo_round.get("winner", "")) if demo_round else ""

        if winner in ("CT", "T"):
            self._winner_seq.append(winner)
            if winner == "CT":
                self._score_ct += 1
            else:
                self._score_t += 1

        kills = self._extract_kills(round_record)
        for k in kills:
            name = str(k.get("attacker", ""))
            if name:
                self._top_kills[name] = self._top_kills.get(name, 0) + 1

        self._maybe_add_turning_point(rn, kills, winner)
        self._last_round_hype = round_hype
        self._update_emotion(winner, round_hype)

    def _update_emotion(self, winner: str, round_hype: float = 0.0) -> None:
        """纯规则解释器：读 hype_rules.json global_accumulation_rules 执行，代码无业务逻辑。"""
        rules = load_hype_rules()
        bias = rules.get("bias", {})
        clamp_cfg = rules.get("global_emotion_clamp", {})
        rule_list = rules.get("global_accumulation_rules", {}).get("rules", [])
        if not rule_list:
            return

        favored_team = str(bias.get("favored_team", "")).strip().upper()
        favored_players = {str(p).strip().lower() for p in bias.get("favored_players", [])}

        favored_won = bool(winner and favored_team and winner.upper() == favored_team)
        favored_lost = bool(winner and favored_team and not favored_won)

        # 连续偏向方输局计数
        consec_losses = 0
        if favored_team and self._winner_seq:
            for w in reversed(self._winner_seq):
                if w.upper() != favored_team:
                    consec_losses += 1
                else:
                    break

        # 偏向选手在本局是否有击杀（从 _top_kills 增量无法判断，留给外部传入；暂以名字命中近似）
        favored_player_killed = bool(
            favored_players and
            any(n.lower() in favored_players for n in self._top_kills)
        )

        def _check(cond, params: dict) -> bool:
            if isinstance(cond, list):
                return all(_check(c, params) for c in cond)
            if cond == "always":
                return True
            if cond == "favored_team_won":
                return favored_won
            if cond == "favored_team_lost":
                return favored_lost
            if cond == "favored_player_got_kill":
                return favored_player_killed
            if cond == "consecutive_favored_losses":
                return consec_losses >= int(params.get("count", 3))
            if cond == "round_hype_gte":
                return round_hype >= float(params.get("value", 0))
            if cond == "round_hype_lt":
                return round_hype < float(params.get("value", 1))
            if cond == "emotion_gte":
                em = str(params.get("emotion", ""))
                val = float(params.get("value", 0))
                return self._emotion_acc.get(em, 0.0) >= val
            return False

        for rule in rule_list:
            if not isinstance(rule, dict):
                continue
            cond = rule.get("condition", "never")
            emotion = str(rule.get("emotion", ""))
            delta = float(rule.get("delta", 0.0))
            params = rule.get("params", {})
            if not emotion or delta == 0.0:
                continue
            if _check(cond, params):
                current = self._emotion_acc.get(emotion, 0.0)
                maximum = float(clamp_cfg.get(emotion, 1.0))
                self._emotion_acc[emotion] = round(max(0.0, min(maximum, current + delta)), 4)

    # ── internal helpers ──

    def _find_demo_round(self, round_no: int, demo_rounds: list[dict] | None) -> dict | None:
        if not demo_rounds:
            return None
        for r in demo_rounds:
            if int(r.get("round_no", 0)) == round_no:
                return r
        return None

    def _extract_kills(self, round_record) -> list[dict]:
        p2 = getattr(round_record, "phase2_vision", None)
        if p2 is None:
            return []
        kills: list[dict] = []
        for kf in (p2.key_frames or []):
            bg = getattr(kf, "background_info", {}) or {}
            kills.extend(bg.get("events", {}).get("kills", []))
        return kills

    def _maybe_add_turning_point(self, round_no: int, kills: list[dict], winner: str) -> None:
        if len(kills) >= 5:
            top: dict[str, int] = {}
            for k in kills:
                name = str(k.get("attacker", ""))
                if name:
                    top[name] = top.get(name, 0) + 1
            for name, cnt in top.items():
                if cnt >= 3:
                    note = f"R{round_no} {name} {cnt}杀"
                    if note not in self._turning_points:
                        self._turning_points.append(note)
        if len(self._turning_points) > 5:
            self._turning_points = self._turning_points[-5:]

    def _is_match_point(self, next_round_no: int) -> bool:
        # CS2 MR12：先到13胜，12胜即赛点（>=13 等于已分胜负，太晚）。
        def _mp(score: int) -> bool:
            if score >= 12:
                return True
            return False
        return _mp(self._score_ct) or _mp(self._score_t)
