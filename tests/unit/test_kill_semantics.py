from sbmachine.kill_semantics import _collateral, _spray_transfer


def test_collateral_does_not_use_z_as_a_map_layer_proxy():
    cfg = {"collateral_tick_tolerance": 2, "one_body_width_units": 32.0}
    kills = [
        {"event_tick": 100, "attacker_pos": [0, 0, 0], "victim_pos": [100, 0, 8000]},
        {"event_tick": 101, "attacker_pos": [0, 0, 0], "victim_pos": [200, 0, -8000]},
    ]
    assert _collateral(kills, cfg) is True


def test_spray_transfer_requires_tightly_grouped_kills():
    cfg = {
        "spray_transfer_min_kills": 2,
        "spray_transfer_max_kill_gap_sec": 1.2,
        "spray_transfer_min_angle_deg": 15.0,
        "spray_transfer_min_ammo_drop": 2,
    }
    kills = [
        {"event_time": 1.0, "weapon": "AK-47", "attacker": "a", "attacker_pos": [0, 0, 0], "victim_pos": [100, 0, 0]},
        {"event_time": 2.3, "weapon": "AK-47", "attacker": "a", "attacker_pos": [0, 0, 0], "victim_pos": [0, 100, 0]},
    ]
    assert _spray_transfer(kills, [], cfg, {"ak-47"}) is False
