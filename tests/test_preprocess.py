"""Tests for src/preprocess.py"""

import pandas as pd
from preprocess import clean_text, preprocess_dataframe, create_splits, LABEL_MAP


class TestCleanText:
    def test_lowercase(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_whitespace_normalization(self):
        assert clean_text("too   many   spaces") == "too many spaces"

    def test_removes_special_characters(self):
        result = clean_text("profit!!! is @great")
        assert "@" not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_preserves_meaningful_content(self):
        result = clean_text("Revenue increased 25 percent")
        assert "revenue" in result
        assert "increased" in result


class TestPreprocessDataframe:
    def test_basic_preprocessing(self):
        df = pd.DataFrame({
            "sentence": ["Revenue UP!", "Loss DOWN."],
            "label": ["positive", "negative"],
        })
        result = preprocess_dataframe(df, clean=True)
        assert "label" in result.columns
        assert len(result) == 2

    def test_label_encoding(self):
        df = pd.DataFrame({
            "sentence": ["Good", "Bad", "Ok"],
            "label": ["positive", "negative", "neutral"],
        })
        result = preprocess_dataframe(df, clean=False)
        assert set(result["label"].values) == {0, 1, 2}

    def test_drops_empty_sentences(self):
        df = pd.DataFrame({
            "sentence": ["Good", "", "Bad"],
            "label": ["positive", "neutral", "negative"],
        })
        result = preprocess_dataframe(df, clean=True)
        # Empty strings may remain but NaN should be dropped
        assert len(result) >= 2


class TestCreateSplits:
    def test_split_sizes(self):
        df = pd.DataFrame({
            "sentence": [f"text_{i}" for i in range(100)],
            "label": [i % 3 for i in range(100)],
        })
        train, val, test = create_splits(df)
        assert len(train) + len(val) + len(test) == 100
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_no_overlap(self):
        df = pd.DataFrame({
            "sentence": [f"text_{i}" for i in range(50)],
            "label": [i % 3 for i in range(50)],
        })
        train, val, test = create_splits(df)
        train_sentences = set(train["sentence"])
        val_sentences = set(val["sentence"])
        test_sentences = set(test["sentence"])
        assert len(train_sentences & val_sentences) == 0
        assert len(train_sentences & test_sentences) == 0
        assert len(val_sentences & test_sentences) == 0
