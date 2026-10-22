import pytest

from app.wer import normalize, word_error_rate


def test_normalize_ignores_case_punctuation_and_notes():
    assert normalize("Hello, World! [music] (laughs) It’s ONNX-Runtime.") == [
        "hello", "world", "it's", "onnx", "runtime",
    ]
    assert normalize("L'acqua è fredda: 3,5 gradi.") == ["l'acqua", "è", "fredda", "3", "5", "gradi"]
    assert normalize("  ...  ") == []


def test_identical_text_is_zero():
    result = word_error_rate("Today we talk about ONNX.", "today we talk about onnx")
    assert result.wer == 0.0 and result.errors == 0 and result.reference_words == 5


def test_counts_each_error_type():
    result = word_error_rate("the cat sat on the mat", "the bat sat on mat today")
    assert (result.substitutions, result.deletions, result.insertions) == (1, 1, 1)
    assert result.wer == pytest.approx(3 / 6)
    assert result.alignment == [
        ["ok", "the", "the"], ["sub", "cat", "bat"], ["ok", "sat", "sat"], ["ok", "on", "on"],
        ["del", "the", ""], ["ok", "mat", "mat"], ["ins", "", "today"],
    ]


def test_empty_cases():
    assert word_error_rate("", "").wer == 0.0
    assert word_error_rate("", "something").wer == 1.0
    assert word_error_rate("two words", "").wer == 1.0
