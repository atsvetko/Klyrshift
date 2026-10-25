"""The keystroke-likelihood test that backs the detector's score path.

``scripts/bench_detector.py`` measures the effect on a held-out word set; these
tests pin the individual cases it fixed, the false corrections it removed, and
the guards it must not loosen.
"""

from __future__ import annotations

import math

import pytest

from klyrshift.config import DEFAULT_HYBRID_CONFIDENCE
from klyrshift.converter import LayoutConverter
from klyrshift.detector import (
    MIN_IMPROVEMENT,
    KeystrokeModel,
    LanguageDetector,
)


@pytest.fixture(scope="module")
def det() -> LanguageDetector:
    """Return a detector with the bundled corpora and no stop words."""
    return LanguageDetector(converter=LayoutConverter(), stop_words=[], min_word_length=3)


# --- the model ---------------------------------------------------------------


def test_counts_are_recovered_from_frequencies() -> None:
    model = KeystrokeModel.from_frequencies({"^a": 0.5, "ab": 0.25, "b$": 0.25})
    assert model is not None
    assert model.counts == {"^a": 2, "ab": 1, "b$": 1}
    assert model.outgoing == {"^": 2, "a": 1, "b": 1}
    assert model.alphabet == frozenset("ab")


def test_foreign_script_pairs_are_dropped() -> None:
    model = KeystrokeModel.from_frequencies({"^д": 0.4, "до": 0.3, "о$": 0.2, "th": 0.1})
    assert model is not None
    assert "th" not in model.counts
    assert model.alphabet == frozenset("до")


def test_empty_model_is_none() -> None:
    assert KeystrokeModel.from_frequencies({}) is None


def test_seen_text_is_more_likely_than_unseen() -> None:
    model = KeystrokeModel.from_frequencies({"^a": 0.5, "ab": 0.25, "b$": 0.25})
    assert model is not None
    assert model.log_likelihood("ab") > model.log_likelihood("ba")
    assert model.log_likelihood("a,b") < model.log_likelihood("ab")


@pytest.mark.parametrize(
    ("text", "layout", "target", "positive"),
    [
        ("ghbdtn", "us", "ru", True),
        ("hello", "us", "ru", False),
        ("руддщ", "ru", "us", True),
        ("привет", "ru", "us", False),
    ],
)
def test_likelihood_margin_sign(
    det: LanguageDetector, text: str, layout: str, target: str, positive: bool
) -> None:
    margin = det.likelihood_margin(text, layout, target)
    assert margin is not None
    assert (margin > 0) is positive


# --- what it fixes -----------------------------------------------------------

# Each was missed on main: vetoed by the plausibility guard, below the score
# threshold, or blocked as a "URL" because a Russian letter sits on ``.``/``[``.
NOW_FIXED = [
    ("компьютер", "ru"),
    ("люди", "ru"),
    ("переключателем", "ru"),
    ("выходных", "ru"),
    ("разъяснения", "ru"),
    ("подъезде", "ru"),
    ("сюда", "ru"),
    ("Чехов", "ru"),
    ("Пушкин", "ru"),
    ("strengths", "us"),
    ("Netflix", "us"),
]


@pytest.mark.parametrize(("word", "layout"), NOW_FIXED)
def test_wrong_layout_word_is_fixed(det: LanguageDetector, word: str, layout: str) -> None:
    other = "us" if layout == "ru" else "ru"
    typed = det.converter.convert(word, layout, other)
    assert det.target_layout(typed, other) == layout
    assert det.should_fix(typed, other) is True


# --- what it no longer breaks ------------------------------------------------

# Correct tokens that main rewrote: a consonant cluster scored as a plausible
# Russian word, or a conversion into fragments too short to score.
NO_LONGER_CHANGED = [
    ("cfg", "us"),
    ("xkb", "us"),
    ("wtf", "us"),
    ("pdf", "us"),
    ("jpg", "us"),
    ("vpn", "us"),
    ("имхо", "ru"),
    ("спб", "ru"),
    ("жкх", "ru"),
]


@pytest.mark.parametrize(("text", "layout"), NO_LONGER_CHANGED)
def test_correct_token_is_left_alone(det: LanguageDetector, text: str, layout: str) -> None:
    assert det.target_layout(text, layout) is None


# A dot or bracket is a letter on the Russian layout, but a real file name, URL
# or code token must keep its structural guard.
STILL_GUARDED = [
    "github.com",
    "file.txt",
    "readme.md",
    "e.g",
    "arr[i]",
    "self.name",
    "list[str]",
    "docs.python.org",
    "user@mail.ru",
    "args[1]",
]


@pytest.mark.parametrize("text", STILL_GUARDED)
def test_structural_tokens_stay_guarded(det: LanguageDetector, text: str) -> None:
    assert det.target_layout(text, "us") is None
    assert det.should_fix(text, "us") is False


def test_separators_are_letters() -> None:
    det = LanguageDetector(converter=LayoutConverter(), stop_words=[])
    assert det.separators_are_letters("k.lb", "us", "ru") is True
    assert det.separators_are_letters("c]tpl", "us", "ru") is True
    assert det.separators_are_letters("user@mail", "us", "ru") is False
    assert det.separators_are_letters("a/b", "us", "ru") is False


def test_likelihood_only_fix_is_withheld_in_hybrid(det: LanguageDetector) -> None:
    # ``strengths`` typed on ru is rejected by the score path and fixed by the
    # likelihood alone. It reports the automatic bar as its confidence, which
    # stays below the default hybrid threshold.
    layout, confidence = det.target_layout_with_confidence("ыекутперы", "ru")
    assert layout == "us"
    assert confidence is not None and not math.isinf(confidence)
    assert confidence == MIN_IMPROVEMENT + det.confidence_threshold
    assert confidence < DEFAULT_HYBRID_CONFIDENCE
