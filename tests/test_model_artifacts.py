"""Guards on committed model artifacts that external users load directly."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import LABEL_MAP_INV  # noqa: E402


def test_finetuned_finbert_config_labels_match_project_mapping():
    """The fine-tuned head was trained on the project's label ids; the config must name them the same way.

    Otherwise a Hugging Face `pipeline()` user gets wrong label names (the app itself decodes with LABEL_MAP_INV).
    """
    config = json.loads((ROOT / "models" / "finbert_finetuned" / "config.json").read_text())
    assert {int(k): v for k, v in config["id2label"].items()} == LABEL_MAP_INV
    assert {k: int(v) for k, v in config["label2id"].items()} == {v: k for k, v in LABEL_MAP_INV.items()}
