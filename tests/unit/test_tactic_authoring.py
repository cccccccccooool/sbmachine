"""人工战术规则填写器单元测试（tests/unit/test_tactic_authoring.py）。

覆盖计划书任务 1–3 的所有测试勾选项：
  任务 1：区域目录和输入解析
  任务 2：分区卡片到 DSL 条件
  任务 3：规则书追加、预览与保存
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

# ── 动态导入工具模块 ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "tactics" / "initialize_tactic_book.py"
SPEC = importlib.util.spec_from_file_location("initialize_tactic_book", SCRIPT)
mod = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(mod)

load_zone_index = mod.load_zone_index
parse_count_range = mod.parse_count_range
zone_condition = mod.zone_condition
event_condition = mod.event_condition
next_rule_id = mod.next_rule_id
build_tactic = mod.build_tactic
preview_tactic = mod.preview_tactic
_save_tactic = mod._save_tactic
_load_existing_book = mod._load_existing_book

# ── 辅助 fixture ──────────────────────────────────────────────────────────────

def _template(callouts: dict) -> dict:
    """构造最小合法地图模板。"""
    return {"version": 2, "map_name": "de_test", "callouts": callouts, "directed_transitions": []}


# ═══════════════════════════════════════════════════════════════════════════════
# 任务 1：区域目录和输入解析
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadZoneIndex:
    def test_sorted_by_casefold(self):
        """区域按 casefold 字母序稳定排列，编号从 1 开始。"""
        tmpl = _template({"B_Main": {"zh": "B区"}, "a_short": {"zh": "A小"}, "Mid": {"zh": "中路"}})
        index = load_zone_index(tmpl)
        ids = [cid for _, cid, _ in index]
        assert ids == sorted(["B_Main", "a_short", "Mid"], key=str.casefold)

    def test_number_starts_at_one(self):
        """编号从 1 开始，连续递增。"""
        tmpl = _template({"X": {"zh": "X区"}, "Y": {"zh": "Y区"}, "Z": {"zh": "Z区"}})
        nums = [num for num, _, _ in load_zone_index(tmpl)]
        assert nums == [1, 2, 3]

    def test_number_maps_to_english_id(self):
        """编号→英文 callout 映射正确。"""
        tmpl = _template({"alpha": {"zh": "阿尔法"}, "beta": {"zh": "贝塔"}})
        index = load_zone_index(tmpl)
        mapping = {num: cid for num, cid, _ in index}
        assert mapping[1] == "alpha"
        assert mapping[2] == "beta"

    def test_missing_zh_returns_empty(self):
        """任一区域缺少 zh 键时 fail-closed 返回空列表。"""
        tmpl = _template({"A": {"zh": "A区"}, "B": {}})
        assert load_zone_index(tmpl) == []

    def test_empty_callouts_returns_empty(self):
        """callouts 为空时返回空列表。"""
        assert load_zone_index(_template({})) == []

    def test_invalid_template_returns_empty(self):
        """非 dict 输入返回空列表。"""
        assert load_zone_index(None) == []  # type: ignore[arg-type]
        assert load_zone_index("bad") == []  # type: ignore[arg-type]

    def test_missing_callouts_key_returns_empty(self):
        """模板缺少 callouts 键时返回空列表。"""
        assert load_zone_index({"version": 2}) == []


class TestParseCountRange:
    def test_exact_number(self):
        assert parse_count_range("1") == [1, 1]
        assert parse_count_range("5") == [5, 5]

    def test_range_with_dash(self):
        assert parse_count_range("3-5") == [3, 5]
        assert parse_count_range("1-1") == [1, 1]

    def test_plus_unbounded(self):
        assert parse_count_range("4+") == [4, None]
        assert parse_count_range("1+") == [1, None]

    def test_whitespace_stripped(self):
        assert parse_count_range(" 2 ") == [2, 2]
        assert parse_count_range(" 2 - 4 ") == [2, 4]

    def test_invalid_returns_none(self):
        assert parse_count_range("abc") is None
        assert parse_count_range("5-3") is None   # 上界小于下界
        assert parse_count_range("") is None
        assert parse_count_range("1.5") is None


# ═══════════════════════════════════════════════════════════════════════════════
# 任务 2：分区卡片到 DSL 条件
# ═══════════════════════════════════════════════════════════════════════════════

class TestZoneCondition:
    def test_no_action_generates_only_zone_count(self):
        """不带动作时只生成一个 zone_count 条件。"""
        cond = zone_condition("T", ["A_Short"], [1, 1])
        assert cond["kind"] == "zone_count"
        assert cond["side"] == "T"
        assert cond["zone"]["callouts_any"] == ["A_Short"]
        assert cond["count"] == [1, 1]

    def test_zone_condition_preserves_count(self):
        """人数范围正确写入 count 字段。"""
        assert zone_condition("CT", ["B_Main"], [3, 5])["count"] == [3, 5]
        assert zone_condition("CT", ["B_Main"], [4, None])["count"] == [4, None]


class TestEventCondition:
    def test_utility_throw_event(self):
        cond = event_condition("utility_throw", "T", ["A_Short"])
        assert cond["kind"] == "event_count"
        assert cond["event"] == "utility_throw"
        assert cond["actor_side"] == "T"
        assert cond["actor_zone"]["callouts_any"] == ["A_Short"]

    def test_kill_event(self):
        cond = event_condition("kill", "T", ["Mid"])
        assert cond["event"] == "kill"

    def test_flash_event(self):
        cond = event_condition("flash", "CT", ["B_Main"])
        assert cond["event"] == "flash"

    def test_event_count_always_one_to_null(self):
        """动作条件固定为至少一次（count=[1,null]）。"""
        cond = event_condition("kill", "T", ["A_Long"])
        assert cond["count"] == [1, None]

    def test_event_bound_to_actor_side_and_zone(self):
        """阵营和分区绑定到用户选择值，不被修改。"""
        cond = event_condition("flash", "CT", ["B_Car"])
        assert cond["actor_side"] == "CT"
        assert cond["actor_zone"]["callouts_any"] == ["B_Car"]

    def test_multiple_zone_conditions_in_order(self):
        """多个分区条件按填写顺序写入 when 数组。"""
        when = [
            zone_condition("T", ["A_Short"], [1, 1]),
            event_condition("utility_throw", "T", ["A_Short"]),
            zone_condition("T", ["B_Main"], [3, 5]),
        ]
        assert when[0]["kind"] == "zone_count"
        assert when[1]["kind"] == "event_count"
        assert when[2]["kind"] == "zone_count"
        assert when[0]["zone"]["callouts_any"] == ["A_Short"]
        assert when[2]["zone"]["callouts_any"] == ["B_Main"]


# ═══════════════════════════════════════════════════════════════════════════════
# 任务 3：规则书追加、预览与保存
# ═══════════════════════════════════════════════════════════════════════════════

class TestNextRuleId:
    def test_empty_book_returns_tactic_001(self):
        assert next_rule_id([]) == "tactic_001"

    def test_increments_past_existing(self):
        existing = [{"id": "tactic_001"}, {"id": "tactic_002"}]
        assert next_rule_id(existing) == "tactic_003"

    def test_fills_gap_above_max(self):
        """取最大编号+1，不填空洞。"""
        existing = [{"id": "tactic_001"}, {"id": "tactic_005"}]
        assert next_rule_id(existing) == "tactic_006"

    def test_ignores_non_numeric_ids(self):
        existing = [{"id": "custom_rule"}, {"id": "tactic_abc"}]
        assert next_rule_id(existing) == "tactic_001"

    def test_zero_padded_to_three_digits(self):
        """编号补零至三位。"""
        assert next_rule_id([{"id": f"tactic_{i:03d}"} for i in range(1, 100)]) == "tactic_100"

    def test_ignores_non_dict_entries(self):
        assert next_rule_id([None, "bad", {"id": "tactic_004"}]) == "tactic_005"  # type: ignore[list-item]


class TestLoadExistingBook:
    def test_missing_file_is_an_empty_book(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "TACTICS_DIR", tmp_path)
        assert _load_existing_book("de_test") == []

    @pytest.mark.parametrize(
        "content",
        ["{bad json", json.dumps({"version": 1}), json.dumps({"tactics": {}}), json.dumps({"tactics": [None]})],
    )
    def test_existing_corrupt_book_is_rejected(self, tmp_path, monkeypatch, content):
        monkeypatch.setattr(mod, "TACTICS_DIR", tmp_path)
        path = tmp_path / "de_test.json"
        path.write_text(content, encoding="utf-8")

        assert _load_existing_book("de_test") is None
        assert path.read_text(encoding="utf-8") == content


def test_script_can_import_sbmachine_with_isolated_python_path(tmp_path):
    code = """
import runpy
import sys
from pathlib import Path

namespace = runpy.run_path(sys.argv[1])
namespace["_save_tactic"].__globals__["TACTICS_DIR"] = Path(sys.argv[2])
tactic = namespace["build_tactic"](
    "tactic_001", "test", "hint", "T",
    [namespace["zone_condition"]("T", ["A"], [1, 1])],
)
raise SystemExit(0 if namespace["_save_tactic"]("de_test", [], tactic) else 1)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(SCRIPT), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "de_test.json").exists()


class TestSaveTactic:
    """_save_tactic() 保存与拒写行为。"""

    def _make_valid_tactic(self, rule_id: str = "tactic_001") -> dict:
        return build_tactic(
            rule_id=rule_id,
            label="测试战术",
            hint="测试提示词",
            side="T",
            when=[zone_condition("T", ["A_Short"], [1, 1])],
        )

    def test_save_creates_file_and_returns_true(self, tmp_path, monkeypatch):
        """校验通过时写入磁盘并返回 True。"""
        monkeypatch.setattr(mod, "TACTICS_DIR", tmp_path)
        tactic = self._make_valid_tactic()
        assert _save_tactic("de_test", [], tactic) is True
        path = tmp_path / "de_test.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["map"] == "de_test"
        assert len(data["tactics"]) == 1

    def test_save_appends_and_preserves_old_order(self, tmp_path, monkeypatch):
        """追加时保留旧规则原顺序。"""
        monkeypatch.setattr(mod, "TACTICS_DIR", tmp_path)
        old = self._make_valid_tactic("tactic_001")
        # 先写入旧规则
        _save_tactic("de_test", [], old)
        new = self._make_valid_tactic("tactic_002")
        _save_tactic("de_test", [old], new)
        data = json.loads((tmp_path / "de_test.json").read_text(encoding="utf-8"))
        assert data["tactics"][0]["id"] == "tactic_001"
        assert data["tactics"][1]["id"] == "tactic_002"

    def test_duplicate_id_refuses_write(self, tmp_path, monkeypatch, capsys):
        """重复 ID 导致编译失败，不写入，返回 False。"""
        monkeypatch.setattr(mod, "TACTICS_DIR", tmp_path)
        existing = [self._make_valid_tactic("tactic_001")]
        duplicate = self._make_valid_tactic("tactic_001")
        result = _save_tactic("de_test", existing, duplicate)
        assert result is False
        captured = capsys.readouterr()
        assert "校验失败" in captured.out

    def test_original_file_unchanged_on_failure(self, tmp_path, monkeypatch):
        """校验失败时原文件内容不被覆盖。"""
        monkeypatch.setattr(mod, "TACTICS_DIR", tmp_path)
        old = self._make_valid_tactic("tactic_001")
        _save_tactic("de_test", [], old)
        original_text = (tmp_path / "de_test.json").read_text(encoding="utf-8")

        duplicate = self._make_valid_tactic("tactic_001")
        _save_tactic("de_test", [old], duplicate)
        assert (tmp_path / "de_test.json").read_text(encoding="utf-8") == original_text

    def test_original_file_unchanged_when_atomic_replace_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "TACTICS_DIR", tmp_path)
        path = tmp_path / "de_test.json"
        original_text = "original"
        path.write_text(original_text, encoding="utf-8")
        tactic = self._make_valid_tactic()
        original_replace = Path.replace

        def fail_tmp_replace(source, target):
            if source.suffix == ".tmp":
                raise OSError("simulated replace failure")
            return original_replace(source, target)

        monkeypatch.setattr(Path, "replace", fail_tmp_replace)
        assert _save_tactic("de_test", [], tactic) is False
        assert path.read_text(encoding="utf-8") == original_text
        assert list(tmp_path.glob("*.tmp")) == []

    def test_optional_matching_fields_are_saved_and_compiled(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mod, "TACTICS_DIR", tmp_path)
        tactic = build_tactic(
            "tactic_001", "中期反清", "开始反清", "CT",
            [zone_condition("CT", ["Mid"], [1, None])],
            scene=["未下包"], time_window_sec=[30.0, 75.0], priority=20,
        )
        assert _save_tactic("de_test", [], tactic) is True
        data = json.loads((tmp_path / "de_test.json").read_text(encoding="utf-8"))
        assert data["tactics"][0]["scene"] == ["未下包"]
        assert data["tactics"][0]["time_window_sec"] == [30.0, 75.0]
        assert data["tactics"][0]["priority"] == 20

    def test_compiled_output_accepted_by_compile_tactic_book(self, tmp_path, monkeypatch):
        """最终输出能被 compile_tactic_book() 编译（tactics 非空）。"""
        monkeypatch.setattr(mod, "TACTICS_DIR", tmp_path)
        from sbmachine.tactic_book import compile_tactic_book
        tactic = self._make_valid_tactic()
        _save_tactic("de_test", [], tactic)
        data = json.loads((tmp_path / "de_test.json").read_text(encoding="utf-8"))
        compiled = compile_tactic_book("de_test", data)
        assert len(compiled.tactics) == 1
        assert compiled.tactics[0].rule_id == "tactic_001"
