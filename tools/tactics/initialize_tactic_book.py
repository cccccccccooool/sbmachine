"""人工战术规则填写器：通过地图区域编号和自然语言输入生成现有战术 DSL JSON。

本工具读取人工地图模板中的区域编号和中文名，将用户输入转换为
`tactic_book.py` 已支持的 `zone_count` 与 `event_count` 条件。
运行时匹配器、LLM 投影和云端数据流不改变。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MAPS_DIR = PROJECT_ROOT / "database" / "maps"
TACTICS_DIR = PROJECT_ROOT / "database" / "tactics"

# 阵营编号映射
_SIDE_MAP: dict[str, str] = {"1": "T", "2": "CT"}
_SIDE_DISPLAY = "1=T  2=CT"

# 动作编号映射（0=无动作）
_ACTION_MAP: dict[str, str] = {"1": "utility_throw", "2": "kill", "3": "flash"}
_ACTION_DISPLAY = "0=不要动作  1=投掷道具  2=击杀  3=闪光"

# 预览用中文动作名
_ACTION_ZH: dict[str, str] = {
    "utility_throw": "投掷道具",
    "kill": "击杀",
    "flash": "闪光",
}


# ── 纯函数 ────────────────────────────────────────────────────────────────────

def load_zone_index(map_template: object) -> list[tuple[int, str, str]]:
    """从地图模板生成稳定有序的区域编号列表：[(编号, 英文ID, 中文名)]。

    任一区域缺少中文名时返回空列表（fail-closed）。
    """
    if not isinstance(map_template, dict):
        return []
    callouts = map_template.get("callouts")
    if not isinstance(callouts, dict) or not callouts:
        return []
    for node in callouts.values():
        if not isinstance(node, dict) or not str(node.get("zh") or "").strip():
            return []
    return [
        (i + 1, cid, str(callouts[cid]["zh"]))
        for i, cid in enumerate(sorted(callouts, key=str.casefold))
    ]


def parse_count_range(raw: str) -> list | None:
    """解析人数范围：'1'→[1,1]、'3-5'→[3,5]、'4+'→[4,null]；格式非法返回 None。"""
    raw = raw.strip()
    if re.fullmatch(r"\d+", raw):
        n = int(raw)
        return [n, n]
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return [lo, hi] if lo <= hi else None
    m = re.fullmatch(r"(\d+)\+", raw)
    if m:
        return [int(m.group(1)), None]
    return None


def zone_condition(side: str, callouts: list[str], count: list) -> dict:
    """生成 zone_count 条件（统计指定阵营在指定区域的存活人数）。"""
    return {
        "kind": "zone_count",
        "side": side,
        "zone": {"callouts_any": list(callouts)},
        "count": count,
    }


def event_condition(event: str, side: str, callouts: list[str]) -> dict:
    """生成 event_count 条件（该阵营在该区至少一人发生过一次指定动作）。"""
    return {
        "kind": "event_count",
        "event": event,
        "actor_side": side,
        "actor_zone": {"callouts_any": list(callouts)},
        "count": [1, None],
    }


def next_rule_id(existing_tactics: list[dict]) -> str:
    """从现有规则提取最大 tactic_数字 编号，返回下一个未占用的 tactic_NNN。"""
    max_num = 0
    for tactic in existing_tactics:
        if not isinstance(tactic, dict):
            continue
        m = re.fullmatch(r"tactic_(\d+)", str(tactic.get("id") or ""))
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"tactic_{max_num + 1:03d}"


def build_tactic(
    rule_id: str,
    label: str,
    hint: str,
    side: str,
    when: list[dict],
    *,
    scene: list[str] | None = None,
    time_window_sec: list[float] | None = None,
    priority: int = 10,
) -> dict:
    """组装单条战术规则 dict，可按需限制场景、回合时间与优先级。"""
    tactic = {"id": rule_id, "label": label, "hint": hint, "side": side, "when": when, "priority": priority}
    if scene is not None:
        tactic["scene"] = scene
    if time_window_sec is not None:
        tactic["time_window_sec"] = time_window_sec
    return tactic


def preview_tactic(tactic: dict, index: list[tuple[int, str, str]]) -> str:
    """生成可读的自然语言预览字符串。"""
    cid_to_zh = {cid: zh for _, cid, zh in index}
    lines = [
        f"  战术：{tactic['label']}",
        f"  提示词：{tactic['hint']}",
        f"  阵营：{tactic['side']}",
        f"  场景：{'、'.join(tactic.get('scene', [])) if tactic.get('scene') else '不限'}",
        f"  回合时间：{tactic['time_window_sec'][0]}–{tactic['time_window_sec'][1]} 秒" if tactic.get("time_window_sec") else "  回合时间：不限",
        f"  优先级：{tactic['priority']}",
        "  条件（AND）：",
    ]
    for cond in tactic["when"]:
        if cond["kind"] == "zone_count":
            zones_zh = "、".join(cid_to_zh.get(c, c) for c in cond["zone"]["callouts_any"])
            lo, hi = cond["count"]
            count_str = str(lo) if lo == hi else (f"{lo}+" if hi is None else f"{lo}–{hi}")
            lines.append(f"    · {cond['side']}方在 {zones_zh} 有 {count_str} 名存活玩家")
        elif cond["kind"] == "event_count":
            zones_zh = "、".join(cid_to_zh.get(c, c) for c in cond["actor_zone"]["callouts_any"])
            ev_zh = _ACTION_ZH.get(cond["event"], cond["event"])
            lines.append(f"    · {cond['actor_side']}方在 {zones_zh} 至少发生过一次{ev_zh}")
    return "\n".join(lines)


# ── 内部辅助（交互 I/O 层）────────────────────────────────────────────────────

def _load_map_template(map_name: str) -> dict | None:
    """加载地图模板；不符合要求时打印原因并返回 None（fail-closed）。"""
    safe = "".join(ch for ch in map_name.lower() if ch.isalnum() or ch in "_-")
    path = MAPS_DIR / f"{safe}.json"
    if not path.exists():
        print(f"地图模板不存在：{path}\n请先运行 initialize_map_template.py 完成地图模板。")
        return None
    try:
        template = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("地图模板读取失败。")
        return None
    index = load_zone_index(template)
    if not index:
        print("地图模板中 callouts 为空或存在缺失中文名的区域，请先补全。")
        return None
    return template


def _load_existing_book(map_name: str) -> list[dict] | None:
    """读取现有规则；仅文件不存在时返回空列表，损坏时 fail-closed。"""
    path = TACTICS_DIR / f"{map_name}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("现有战术书读取失败，已拒绝继续，避免覆盖原文件。")
        return None
    if not isinstance(data, dict):
        print("现有战术书格式损坏，已拒绝继续，避免覆盖原文件。")
        return None
    tactics = data.get("tactics")
    if not isinstance(tactics, list) or not all(isinstance(item, dict) for item in tactics):
        print("现有战术书格式损坏，已拒绝继续，避免覆盖原文件。")
        return None
    return tactics


def _collect_zones(
    index: list[tuple[int, str, str]], side: str
) -> list[dict] | None:
    """分区循环：返回 when 条件列表（至少一个分区），用户输入 n 放弃时返回 None。"""
    num_to_cid = {num: cid for num, cid, _ in index}
    num_to_zh = {num: zh for num, _, zh in index}
    when: list[dict] = []

    while True:
        print("\n可用区域：")
        for num, cid, zh in index:
            print(f"  {num}. {zh}（{cid}）")
        raw = input("\n  选择分区编号（输入 q 放弃本次战术）：").strip()
        if raw.casefold() in {"q", "quit", "放弃"}:
            return None
        try:
            zone_num = int(raw)
        except ValueError:
            print("  请输入数字编号。")
            continue
        if zone_num not in num_to_cid:
            print("  编号不在列表中。")
            continue

        callout_id = num_to_cid[zone_num]
        zh_name = num_to_zh[zone_num]

        # 人数范围
        while True:
            count_raw = input(f"  {zh_name} 人数范围（例如 1、3-5、4+）：").strip()
            count = parse_count_range(count_raw)
            if count is not None:
                break
            print("  格式不合法，请用 1 / 3-5 / 4+ 格式。")

        when.append(zone_condition(side, [callout_id], count))

        # 动作条件
        print(f"  动作条件：{_ACTION_DISPLAY}")
        while True:
            action_raw = input("  选择动作（回车或 0 表示不要）：").strip() or "0"
            if action_raw == "0":
                break
            event = _ACTION_MAP.get(action_raw)
            if event is not None:
                when.append(event_condition(event, side, [callout_id]))
                break
            print("  无效编号，请输入 0–3。")

        if input("\n  继续添加分区？[y/N]：").strip().casefold() not in {"y", "yes", "是"}:
            break

    return when


def _save_tactic(map_name: str, existing: list[dict], new_tactic: dict) -> bool:
    """将新战术追加到规则书并保存；保存前经 compile_tactic_book() 校验。

    校验通过时写入磁盘并返回 True；校验失败时打印原因、不写入、返回 False。
    """
    from sbmachine.tactic_book import compile_tactic_book  # 延迟导入，避免工具层循环依赖

    merged = list(existing) + [new_tactic]
    source = {"version": 1, "map": map_name, "tactics": merged}
    compiled = compile_tactic_book(map_name, source)
    if not compiled.tactics:
        print("\n[校验失败] 生成的规则书无法通过 compile_tactic_book()，写入已取消。")
        print("  常见原因：重复 ID、非法字段或条件格式错误。")
        return False

    TACTICS_DIR.mkdir(parents=True, exist_ok=True)
    path = TACTICS_DIR / f"{map_name}.json"
    content = json.dumps(source, ensure_ascii=False, indent=2)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    except OSError as exc:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        print(f"\n[保存失败] 原文件未修改：{exc}")
        return False
    print(f"\n[OK] 已保存至 {path}")
    return True


# ── 入口 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="人工战术规则填写器")
    parser.add_argument("--map", metavar="MAP_NAME", default="", help="地图名称（可选，省略则交互输入）")
    args = parser.parse_args()

    # ── 地图名（统一小写+过滤，与 _load_map_template 的 safe 路径保持一致）──────
    raw_map = args.map.strip()
    while not raw_map:
        raw_map = input("地图名称（例如 de_ancient）：").strip()
    map_name = "".join(ch for ch in raw_map.lower() if ch.isalnum() or ch in "_-")

    template = _load_map_template(map_name)
    if template is None:
        sys.exit(1)

    index = load_zone_index(template)
    existing = _load_existing_book(map_name)
    if existing is None:
        sys.exit(1)

    if existing:
        print(f"\n已找到 {len(existing)} 条现有战术规则，新规则将追加。")

    # ── 主循环 ────────────────────────────────────────────────────────────────
    while True:
        print("\n" + "─" * 48)

        # 战术名
        label = ""
        while not label:
            label = input("战术名（例如 假爆A真打B）：").strip()

        # 提示词（为空则与战术名相同）
        hint = input(f"解说提示词（回车使用战术名）：").strip() or label

        # 阵营
        side = ""
        while side not in _SIDE_MAP.values():
            raw = input(f"阵营 [{_SIDE_DISPLAY}]：").strip()
            side = _SIDE_MAP.get(raw, "")
            if not side:
                print("  请输入 1 或 2。")

        # 可选匹配范围
        scene_raw = input("适用场景（多个用逗号分隔，回车不限）：").strip()
        scenes = [item.strip() for item in re.split(r"[,，]", scene_raw) if item.strip()] or None

        while True:
            time_raw = input("回合时间范围秒（例如 30-75，回车不限）：").strip()
            if not time_raw:
                time_window = None
                break
            parsed_time = parse_count_range(time_raw)
            if parsed_time is not None and parsed_time[1] is not None:
                time_window = [float(parsed_time[0]), float(parsed_time[1])]
                break
            print("  格式不合法，请输入起止秒，例如 30-75。")

        while True:
            priority_raw = input("优先级 [10]：").strip()
            try:
                priority = int(priority_raw) if priority_raw else 10
                break
            except ValueError:
                print("  优先级必须是整数。")

        # 分区
        when = _collect_zones(index, side)
        if when is None:
            print("  已放弃本次战术。")
            if input("\n继续填写新战术？[Y/n]：").strip().casefold() in {"n", "no", "否"}:
                break
            continue

        # 生成规则 ID
        rule_id = next_rule_id(existing)
        tactic = build_tactic(
            rule_id, label, hint, side, when,
            scene=scenes, time_window_sec=time_window, priority=priority,
        )

        # 预览
        print("\n─── 规则预览 ─────────────────────────────────────")
        print(preview_tactic(tactic, index))
        print(f"  规则 ID：{rule_id}")
        print("──────────────────────────────────────────────────")

        action = ""
        while action not in {"s", "a", "q"}:
            action = input("  s=保存  a=放弃本条  q=退出程序：").strip().casefold()

        if action == "q":
            break
        if action == "a":
            print("  已放弃本条战术，原文件不变。")
        else:
            saved = _save_tactic(map_name, existing, tactic)
            if saved:
                # 更新内存中的已存在规则，下一条规则 ID 不重复
                existing = existing + [tactic]

        if input("\n继续填写新战术？[Y/n]：").strip().casefold() in {"n", "no", "否"}:
            break

    print("\n已退出战术填写器。")


if __name__ == "__main__":
    main()
