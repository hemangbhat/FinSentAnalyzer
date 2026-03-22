"""Tests for src/utils.py"""

import logging
from pathlib import Path

from utils import (
    get_project_root,
    get_model_dir,
    get_data_dir,
    get_results_dir,
    setup_logging,
    LABEL_MAP,
    LABEL_MAP_INV,
    BASELINE_CLASSIFIERS,
    TRANSFORMER_MODELS,
    get_model_info,
)


class TestGetProjectRoot:
    def test_returns_path(self):
        root = get_project_root()
        assert isinstance(root, Path)

    def test_contains_src_dir(self):
        root = get_project_root()
        assert (root / "src").exists()

    def test_contains_app_dir(self):
        root = get_project_root()
        assert (root / "app").exists()


class TestPathHelpers:
    def test_get_model_dir(self):
        model_dir = get_model_dir()
        assert isinstance(model_dir, Path)
        assert model_dir.name == "models"
        assert model_dir.exists()

    def test_get_data_dir(self):
        data_dir = get_data_dir()
        assert isinstance(data_dir, Path)
        assert data_dir.name == "data"

    def test_get_data_dir_with_subdir(self):
        data_dir = get_data_dir("processed")
        assert data_dir.name == "processed"

    def test_get_results_dir(self):
        results_dir = get_results_dir()
        assert isinstance(results_dir, Path)
        assert results_dir.name == "results"


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging("test_logger")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self):
        logger = setup_logging("test_handler_logger")
        assert len(logger.handlers) > 0

    def test_logger_level_is_info(self):
        logger = setup_logging("test_level_logger")
        assert logger.level == logging.INFO

    def test_no_duplicate_handlers(self):
        name = "test_dedup_logger"
        logger1 = setup_logging(name)
        handler_count = len(logger1.handlers)
        logger2 = setup_logging(name)
        assert len(logger2.handlers) == handler_count


class TestConstants:
    def test_label_map_has_three_entries(self):
        assert len(LABEL_MAP) == 3

    def test_label_map_values(self):
        assert LABEL_MAP["negative"] == 0
        assert LABEL_MAP["neutral"] == 1
        assert LABEL_MAP["positive"] == 2

    def test_label_map_inv_roundtrips(self):
        for label, idx in LABEL_MAP.items():
            assert LABEL_MAP_INV[idx] == label

    def test_baseline_classifiers(self):
        assert "logreg" in BASELINE_CLASSIFIERS
        assert "svm" in BASELINE_CLASSIFIERS
        assert len(BASELINE_CLASSIFIERS) >= 6

    def test_transformer_models(self):
        assert "finbert" in TRANSFORMER_MODELS


class TestGetModelInfo:
    def test_known_model_returns_dict(self):
        info = get_model_info("baseline_svm")
        assert isinstance(info, dict)
        assert "name" in info
        assert "accuracy" in info

    def test_unknown_model_returns_defaults(self):
        info = get_model_info("nonexistent_model_xyz")
        assert info["name"] == "nonexistent_model_xyz"
        assert info["accuracy"] == "N/A"

    def test_finbert_pretrained_info(self):
        info = get_model_info("finbert_pretrained")
        assert "FinBERT" in info["name"]
        assert "Transformer" in info["type"]
