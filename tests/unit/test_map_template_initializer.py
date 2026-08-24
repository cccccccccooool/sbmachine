import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from sbmachine import spatial_context


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "maps" / "initialize_map_template.py"
SPEC = importlib.util.spec_from_file_location("initialize_map_template", SCRIPT)
initializer = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(initializer)


def test_collects_exact_callouts_from_parser_ticks_and_preserves_manual_metadata(tmp_path):
    ticks = tmp_path / "ticks.jsonl"
    ticks.write_text(
        "\n".join([
            json.dumps({"callout": "Lobby", "x": 0, "y": 0, "z": 0}),
            json.dumps({"callout": "BombsiteB", "x": 0, "y": 0, "z": -700}),
            json.dumps({"callout": "Lobby", "x": 0, "y": 0, "z": 0}),
        ]),
    )

    assert list(initializer.collect_callout_stats([ticks])) == ["BombsiteB", "Lobby"]
    template = initializer.new_manual_template(
        "de_nuke",
        ["BombsiteB", "Lobby"],
        labels={"BombsiteB": "B包点（下层）", "Lobby": "大厅"},
    )
    template["callouts"]["BombsiteB"].update({"layer": "lower", "level": -1})
    template["directed_transitions"].append({
        "from": "Lobby", "to": "BombsiteB", "kind": "vent", "level_delta": -1,
        "one_way": False, "samples": 0, "source": "manual",
    })

    template["source"]["manual_reviewed"] = True
    assert initializer.validate_template(template) == []
    assert template["callouts"]["BombsiteB"]["zh"] == "B包点（下层）"


def test_template_validation_rejects_unreviewed_or_untranslated_bootstrap():
    template = initializer.new_manual_template("de_nuke", ["Lobby"])
    errors = initializer.validate_template(template)
    assert any("manual_reviewed" in error for error in errors)
    assert any("Chinese label" in error for error in errors)

def test_collect_chinese_labels_reprompts_when_default_is_english(monkeypatch):
    template = initializer.new_manual_template("de_nuke", ["Lobby"])
    answers = iter(["", "大厅"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    initializer._collect_chinese_labels(template)

    assert template["callouts"]["Lobby"]["zh"] == "大厅"


def test_separate_layers_are_not_nearby_without_a_declared_vertical_connection(monkeypatch):
    template = {
        "version": 2,
        "source": {"manual_review_required": True, "manual_reviewed": True},
        "callouts": {
            "Upper": {"zh": "上层", "layer": "upper", "level": 1},
            "Lower": {"zh": "下层", "layer": "lower", "level": -1},
        },
        "directed_transitions": [],
    }
    monkeypatch.setattr(spatial_context, "load_map_template", lambda _: template)
    players = [
        {"name": "pov", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 0, "callout": "Upper"},
        {"name": "lower_enemy", "side": "T", "hp": 100, "x": 10, "y": 10, "z": -10, "callout": "Lower"},
    ]
    frame = {"who": {"pov_player": "pov"}, "where": {"players": players}, "what": {"desc": ""}}

    result = spatial_context.resolve_spatial_context("de_nuke", "未下包", [frame], [])
    assert result["nearby"]["enemies"] == []


def test_declared_vertical_connection_is_exposed_as_vertical_relation(monkeypatch):
    template = {
        "version": 2,
        "source": {"manual_review_required": True, "manual_reviewed": True},
        "callouts": {
            "Upper": {"zh": "上层", "layer": "upper", "level": 1},
            "Lower": {"zh": "下层", "layer": "lower", "level": -1},
        },
        "directed_transitions": [{
            "from": "Upper", "to": "Lower", "kind": "stairs", "level_delta": -2,
            "one_way": False, "samples": 0, "source": "manual",
        }],
    }
    monkeypatch.setattr(spatial_context, "load_map_template", lambda _: template)
    players = [
        {"name": "pov", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 0, "callout": "Upper"},
        {"name": "lower_enemy", "side": "T", "hp": 100, "x": 10, "y": 10, "z": -10, "callout": "Lower"},
    ]
    frame = {"who": {"pov_player": "pov"}, "where": {"players": players}, "what": {"desc": ""}}

    result = spatial_context.resolve_spatial_context("de_nuke", "未下包", [frame], [])
    assert result["nearby"]["enemies"][0]["relation"] == "vertical_connected"
    assert result["nearby"]["enemies"][0]["callout_zh"] == "下层"
    assert spatial_context._callout_relation(template, "Lower", "Upper") == "vertical_connected"


def test_one_way_edge_is_not_matched_in_reverse():
    template = {
        "version": 2,
        "source": {"manual_review_required": True, "manual_reviewed": True},
        "callouts": {
            "Upper": {"zh": "上层", "layer": "upper", "level": 1},
            "Lower": {"zh": "下层", "layer": "lower", "level": -1},
        },
        "directed_transitions": [{
            "from": "Upper", "to": "Lower", "kind": "drop", "level_delta": -2,
            "one_way": True, "source": "manual",
        }],
    }

    assert spatial_context._callout_relation(template, "Upper", "Lower") == "vertical_connected"
    assert spatial_context._callout_relation(template, "Lower", "Upper") == "separate_level"


def test_unknown_callouts_never_enter_nearby(monkeypatch):
    template = {
        "version": 2,
        "source": {"manual_review_required": True, "manual_reviewed": True},
        "callouts": {"Known": {"zh": "已知", "layer": "L2", "level": 2}},
        "directed_transitions": [],
    }
    monkeypatch.setattr(spatial_context, "load_map_template", lambda _: template)
    players = [
        {"name": "pov", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 0, "callout": "Unknown"},
        {"name": "enemy", "side": "T", "hp": 100, "x": 10, "y": 0, "z": 0, "callout": "Unknown"},
    ]
    frame = {"who": {"pov_player": "pov"}, "where": {"players": players}, "what": {"desc": ""}}

    result = spatial_context.resolve_spatial_context("de_test", "未下包", [frame], [])
    assert result["nearby"]["enemies"] == []


def test_calibration_keeps_manual_chinese_layer_and_connector(tmp_path):
    builder_spec = importlib.util.spec_from_file_location("build_map_template", ROOT / "tools" / "maps" / "build_map_template.py")
    builder = importlib.util.module_from_spec(builder_spec)
    assert builder_spec and builder_spec.loader
    builder_spec.loader.exec_module(builder)
    ticks = tmp_path / "ticks.jsonl"
    ticks.write_text(
        "\n".join([
            json.dumps({"tick": 1, "steamid": "1", "callout": "Lobby", "x": 10, "y": 20, "z": 30}),
            json.dumps({"tick": 2, "steamid": "1", "callout": "BombsiteB", "x": 20, "y": 30, "z": -700}),
        ]),
    )
    base = initializer.new_manual_template("de_nuke", ["Lobby", "BombsiteB"], labels={"Lobby": "大厅", "BombsiteB": "B包点"})
    base["source"]["manual_reviewed"] = True
    base["callouts"]["BombsiteB"].update({"layer": "lower", "level": -1})
    base["directed_transitions"] = [{
        "from": "Lobby", "to": "BombsiteB", "kind": "vent", "level_delta": -1,
        "one_way": True, "samples": 0, "source": "manual",
    }]

    result = builder.build_template([ticks], "de_nuke", base_template=base)
    assert result["version"] == 2
    assert result["callouts"]["BombsiteB"]["zh"] == "B包点"
    assert result["callouts"]["BombsiteB"]["level"] == -1
    assert result["directed_transitions"][0]["kind"] == "vent"
    assert result["directed_transitions"][0]["samples"] == 1

def test_calibration_never_creates_observed_only_edges(tmp_path):
    builder_spec = importlib.util.spec_from_file_location("build_map_template_no_auto_edges", ROOT / "tools" / "maps" / "build_map_template.py")
    builder = importlib.util.module_from_spec(builder_spec)
    assert builder_spec and builder_spec.loader
    builder_spec.loader.exec_module(builder)
    ticks = tmp_path / "ticks.jsonl"
    ticks.write_text("\n".join([
        json.dumps({"tick": 1, "steamid": "1", "callout": "A", "x": 0, "y": 0, "z": 0}),
        json.dumps({"tick": 2, "steamid": "1", "callout": "B", "x": 10, "y": 0, "z": 0}),
    ]), encoding="utf-8")
    base = initializer.new_manual_template("de_test", ["A", "B"], labels={"A": "甲点", "B": "乙点"})
    base["source"]["manual_reviewed"] = True

    result = builder.build_template([ticks], "de_test", base_template=base)

    assert result["directed_transitions"] == []

def test_new_templates_default_level_is_one_and_unreviewed():
    template = initializer.new_manual_template("de_train", ["Upper", "Lower"])
    assert template["callouts"]["Upper"]["level"] == 1
    assert template["callouts"]["Upper"]["layer"] == "L1"
    assert template["source"]["manual_reviewed"] is False
    assert spatial_context._is_reviewed_template(template) is False

def test_coordinate_fallback_does_not_publish_xy_nearby_candidates(monkeypatch):
    monkeypatch.setattr(spatial_context, "load_map_template", lambda _: {})
    players = [
        {"name": "pov", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 8000, "callout": "A"},
        {"name": "near", "side": "T", "hp": 100, "x": 20, "y": 0, "z": -8000, "callout": "B"},
    ]
    frame = {"who": {"pov_player": "pov"}, "where": {"players": players}, "what": {"desc": ""}}

    result = spatial_context.resolve_spatial_context("de_nuke", "未下包", [frame], [])
    assert result["map_precision"] == "coordinate_fallback"
    assert result["nearby"] == {"teammates": [], "enemies": []}

def test_reviewed_graph_excludes_same_level_unlinked_callouts(monkeypatch):
    template = {
        "version": 2,
        "source": {"manual_review_required": True, "manual_reviewed": True},
        "callouts": {
            "A": {"zh": "甲", "layer": "L2", "level": 2},
            "B": {"zh": "乙", "layer": "L2", "level": 2},
        },
        "directed_transitions": [],
    }
    monkeypatch.setattr(spatial_context, "load_map_template", lambda _: template)
    players = [
        {"name": "pov", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 0, "callout": "A"},
        {"name": "through_wall", "side": "T", "hp": 100, "x": 20, "y": 0, "z": 0, "callout": "B"},
    ]
    frame = {"who": {"pov_player": "pov"}, "where": {"players": players}, "what": {"desc": ""}}

    result = spatial_context.resolve_spatial_context("de_nuke", "未下包", [frame], [])
    assert result["map_precision"] == "reviewed_graph"
    assert result["nearby"]["enemies"] == []


def test_observed_transition_is_not_a_manual_spatial_edge(monkeypatch):
    template = {
        "version": 2,
        "source": {"manual_review_required": True, "manual_reviewed": True},
        "callouts": {"A": {"zh": "甲", "layer": "L2", "level": 2}, "B": {"zh": "乙", "layer": "L2", "level": 2}},
        "directed_transitions": [{"from": "A", "to": "B", "kind": "walk", "level_delta": 0, "one_way": False, "source": "observed"}],
    }
    monkeypatch.setattr(spatial_context, "load_map_template", lambda _: template)
    players = [
        {"name": "pov", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 0, "callout": "A"},
        {"name": "observed_only", "side": "T", "hp": 100, "x": 20, "y": 0, "z": 0, "callout": "B"},
    ]
    frame = {"who": {"pov_player": "pov"}, "where": {"players": players}, "what": {"desc": ""}}

    assert spatial_context.resolve_spatial_context("de_nuke", "未下包", [frame], [])["nearby"]["enemies"] == []

def test_cli_catalog_uses_only_parser_observed_callouts(tmp_path):
    ticks = tmp_path / "ticks.jsonl"
    ticks.write_text(json.dumps({"callout": "ParserOnly", "x": 0, "y": 0, "z": 0}) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--map", "de_nuke", "--ticks", str(ticks), "--show-catalog"],
        check=False,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout.decode("utf-8", errors="ignore")
    assert "ParserOnly" in output
    assert "Lobby" not in output


# ── 任务 1：默认层级改为 1 ──────────────────────────────────────────────────

def test_template_validates_zero_and_negative_levels():
    template = initializer.new_manual_template("de_nuke", ["A", "B"], labels={"A": "甲点", "B": "乙点"})
    template["callouts"]["A"]["level"] = 0
    template["callouts"]["A"]["layer"] = "L0"
    template["callouts"]["B"]["level"] = -3
    template["callouts"]["B"]["layer"] = "L-3"
    template["source"]["manual_reviewed"] = True
    assert initializer.validate_template(template) == []


def test_validate_template_rejects_self_loop():
    template = initializer.new_manual_template("de_test", ["A"], labels={"A": "甲"})
    template["source"]["manual_reviewed"] = True
    template["directed_transitions"] = [{
        "from": "A", "to": "A", "kind": "walk", "level_delta": 0,
        "one_way": False, "samples": 0, "source": "manual",
    }]
    errors = initializer.validate_template(template)
    assert any("self-loop" in e for e in errors)


# ── 任务 2：三轮交互纯函数 ──────────────────────────────────────────────────

def test_make_node_index_sorted_by_casefold():
    callouts = {"BombsiteB": {"zh": "B包点"}, "Lobby": {"zh": "大厅"}, "aShort": {"zh": "A小道"}}
    index = initializer._make_node_index(callouts)
    assert index[0] == (1, "aShort", "A小道")
    assert index[1] == (2, "BombsiteB", "B包点")
    assert index[2] == (3, "Lobby", "大厅")


def test_make_node_index_number_to_id_lookup():
    callouts = {"A": {"zh": "甲"}, "B": {"zh": "乙"}}
    index = initializer._make_node_index(callouts)
    num_to_cid = {num: cid for num, cid, _ in index}
    assert num_to_cid[1] == "A"
    assert num_to_cid[2] == "B"


def test_add_edge_non_one_way_generates_reverse():
    template = initializer.new_manual_template("de_test", ["A", "B"], labels={"A": "甲", "B": "乙"})
    initializer._add_edge(template, "A", "B", "walk", False, 0)
    assert len(template["directed_transitions"]) == 2
    pairs = {(e["from"], e["to"]) for e in template["directed_transitions"]}
    assert pairs == {("A", "B"), ("B", "A")}


def test_add_edge_one_way_drop_no_reverse():
    template = initializer.new_manual_template("de_test", ["A", "B"], labels={"A": "甲", "B": "乙"})
    initializer._add_edge(template, "A", "B", "drop", True, -2)
    assert len(template["directed_transitions"]) == 1
    assert template["directed_transitions"][0]["from"] == "A"
    assert template["directed_transitions"][0]["to"] == "B"


def test_configure_edges_reprompts_invalid_one_way_input(monkeypatch):
    template = initializer.new_manual_template("de_test", ["A", "B"], labels={"A": "甲", "B": "乙"})
    answers = iter(["y", "1", "2", "1", "abc", "n", "", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    initializer._configure_edges(template)

    assert len(template["directed_transitions"]) == 2
    assert all(edge["one_way"] is False for edge in template["directed_transitions"])


def test_add_edge_reverse_has_negated_level_delta():
    template = initializer.new_manual_template("de_test", ["A", "B"], labels={"A": "甲", "B": "乙"})
    initializer._add_edge(template, "A", "B", "stairs", False, -1)
    fwd = next(e for e in template["directed_transitions"] if e["from"] == "A")
    rev = next(e for e in template["directed_transitions"] if e["from"] == "B")
    assert fwd["level_delta"] == -1
    assert rev["level_delta"] == 1


# ── 任务 3：构建器边界 ──────────────────────────────────────────────────────

def test_build_preserves_drop_one_way_and_level_delta(tmp_path):
    builder_spec = importlib.util.spec_from_file_location(
        "build_map_template_drop", ROOT / "tools" / "maps" / "build_map_template.py"
    )
    builder = importlib.util.module_from_spec(builder_spec)
    assert builder_spec and builder_spec.loader
    builder_spec.loader.exec_module(builder)
    ticks = tmp_path / "ticks.jsonl"
    ticks.write_text(
        "\n".join([
            json.dumps({"tick": 1, "steamid": "s1", "callout": "Top", "x": 0, "y": 0, "z": 300}),
            json.dumps({"tick": 2, "steamid": "s1", "callout": "Bottom", "x": 0, "y": 0, "z": -200}),
        ]),
        encoding="utf-8",
    )
    base = initializer.new_manual_template(
        "de_nuke", ["Top", "Bottom"], labels={"Top": "顶部", "Bottom": "底部"}
    )
    base["source"]["manual_reviewed"] = True
    base["callouts"]["Top"]["level"] = 2
    base["callouts"]["Bottom"]["level"] = 1
    base["directed_transitions"] = [{
        "from": "Top", "to": "Bottom", "kind": "drop",
        "level_delta": -1, "one_way": True, "samples": 0, "source": "manual",
    }]
    result = builder.build_template([ticks], "de_nuke", base_template=base)
    edge = result["directed_transitions"][0]
    assert edge["kind"] == "drop"
    assert edge["one_way"] is True
    assert edge["level_delta"] == -1


def test_build_edge_kind_does_not_affect_transition_count(tmp_path):
    builder_spec = importlib.util.spec_from_file_location(
        "build_map_template_kind", ROOT / "tools" / "maps" / "build_map_template.py"
    )
    builder = importlib.util.module_from_spec(builder_spec)
    assert builder_spec and builder_spec.loader
    builder_spec.loader.exec_module(builder)
    ticks = tmp_path / "ticks.jsonl"
    ticks.write_text(
        "\n".join([
            json.dumps({"tick": 1, "steamid": "s1", "callout": "X", "x": 0, "y": 0, "z": 0}),
            json.dumps({"tick": 2, "steamid": "s1", "callout": "Y", "x": 10, "y": 0, "z": 0}),
        ]),
        encoding="utf-8",
    )
    for kind in ("walk", "stairs", "ladder", "ramp", "vent"):
        base = initializer.new_manual_template("de_test", ["X", "Y"], labels={"X": "甲", "Y": "乙"})
        base["source"]["manual_reviewed"] = True
        base["directed_transitions"] = [{
            "from": "X", "to": "Y", "kind": kind,
            "level_delta": 0, "one_way": False, "samples": 0, "source": "manual",
        }]
        result = builder.build_template([ticks], "de_test", base_template=base)
        assert result["directed_transitions"][0]["samples"] == 1, f"kind={kind} broke transition count"


# ── 任务 4：空间语义运行时 ─────────────────────────────────────────────────

def test_nuke_third_floor_drop_is_vertical_connected_forward_only(monkeypatch):
    """三楼→A包 drop（单向）：正向 vertical_connected，反向不产生邻接。"""
    template = {
        "version": 2,
        "source": {"manual_review_required": True, "manual_reviewed": True},
        "callouts": {
            "ThirdFloor": {"zh": "三楼", "layer": "L3", "level": 3},
            "ASite": {"zh": "A包点", "layer": "L1", "level": 1},
        },
        "directed_transitions": [{
            "from": "ThirdFloor", "to": "ASite", "kind": "drop",
            "level_delta": -2, "one_way": True, "samples": 0, "source": "manual",
        }],
    }
    monkeypatch.setattr(spatial_context, "load_map_template", lambda _: template)

    players_top = [
        {"name": "pov", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 300, "callout": "ThirdFloor"},
        {"name": "enemy", "side": "T", "hp": 100, "x": 0, "y": 0, "z": 0, "callout": "ASite"},
    ]
    frame_top = {"who": {"pov_player": "pov"}, "where": {"players": players_top}, "what": {"desc": ""}}
    result = spatial_context.resolve_spatial_context("de_nuke", "未下包", [frame_top], [])
    assert result["nearby"]["enemies"][0]["relation"] == "vertical_connected"

    players_bot = [
        {"name": "pov", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 0, "callout": "ASite"},
        {"name": "enemy", "side": "T", "hp": 100, "x": 0, "y": 0, "z": 300, "callout": "ThirdFloor"},
    ]
    frame_bot = {"who": {"pov_player": "pov"}, "where": {"players": players_bot}, "what": {"desc": ""}}
    result2 = spatial_context.resolve_spatial_context("de_nuke", "未下包", [frame_bot], [])
    assert result2["nearby"]["enemies"] == []


def test_sight_only_no_edge_does_not_produce_adjacency(monkeypatch):
    """视野相通但无移动边的区域不应产生邻接关系。"""
    template = {
        "version": 2,
        "source": {"manual_review_required": True, "manual_reviewed": True},
        "callouts": {
            "Window": {"zh": "窗口", "layer": "L2", "level": 2},
            "Pit": {"zh": "坑", "layer": "L2", "level": 2},
        },
        "directed_transitions": [],
    }
    monkeypatch.setattr(spatial_context, "load_map_template", lambda _: template)
    players = [
        {"name": "pov", "side": "CT", "hp": 100, "x": 0, "y": 0, "z": 0, "callout": "Window"},
        {"name": "enemy", "side": "T", "hp": 100, "x": 50, "y": 0, "z": 0, "callout": "Pit"},
    ]
    frame = {"who": {"pov_player": "pov"}, "where": {"players": players}, "what": {"desc": ""}}
    result = spatial_context.resolve_spatial_context("de_nuke", "未下包", [frame], [])
    assert result["nearby"]["enemies"] == []
