import json

import pytest

from sbmachine.preflight import (
    PublishContractError,
    validate_llmb_draft_package,
    validate_render_package,
)
from tests.support.phase4_v2 import b_package, c_package


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_b_v2_and_passthrough_c_v2_validate_together(tmp_path):
    b_payload = b_package()
    c_payload = c_package(b_payload, integration_status="llmb_passthrough", final_text="B draft text")
    b_path = _write(tmp_path / "b.json", b_payload)
    c_path = _write(tmp_path / "c.json", c_payload)

    validate_llmb_draft_package(b_path)
    validate_render_package(c_path, b_path)


def test_c_v2_final_text_is_not_required_in_b_source_block(tmp_path):
    b_payload = b_package()
    c_payload = c_package(b_payload, final_text="C owns this final text")
    c_path = _write(tmp_path / "c.json", c_payload)

    validate_render_package(c_path)
    assert c_payload["rounds"][0]["render_units"][0]["final_text"] == "C owns this final text"


def test_c_v2_artifact_identity_is_fail_closed(tmp_path):
    b_payload = b_package()
    c_payload = c_package(b_payload)
    c_payload["artifact_identity"] = "tampered"
    with pytest.raises(PublishContractError, match="artifact_identity"):
        validate_render_package(_write(tmp_path / "c.json", c_payload))

