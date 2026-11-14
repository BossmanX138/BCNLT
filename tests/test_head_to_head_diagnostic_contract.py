import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_bcnlt_head_to_head_mk2.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("bcnlt_head_to_head", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_head_to_head_report_does_not_claim_semantic_winner():
    module = _load_script()
    model = {
        "standard_extractor_target_hit_rate": 0.5,
        "bcnlt_extractor_target_hit_rate": 1.0,
        "extractor_target_hit_rate_gain": 0.5,
        "readiness_protocol_pass": True,
        "latency_seconds": {"mean": 0.1},
    }
    payload = {
        "file_count": 1,
        "probe_count": 2,
        "models": {"wizard_phi_v02": model, "runeforge_mk2": model},
        "paired_extractor_target_outcomes": {
            "both_hit": 2,
            "wizard_only_hit": 0,
            "runeforge_only_hit": 0,
            "neither_hit": 0,
        },
        "diagnostic_leader": "Tie",
        "semantic_winner": None,
        "independent_semantic_oracle": False,
        "promotion_eligible": False,
    }

    markdown = module.render_markdown(payload)

    assert "Semantic winner: **not established**" in markdown
    assert "Extractor-target hit rate" in markdown
    assert "Winner by BCNLT-context accuracy" not in markdown
    assert "not promotion evidence" in markdown
