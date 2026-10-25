"""The held-out detector benchmark (``scripts/bench_detector.py``) stays usable.

The benchmark is only meaningful while its word list is well formed: every word
is typed in a known layout, its wrong-layout rendering converts back exactly, and
no word is listed twice. These checks keep a hand edit of the TSV honest.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from klyrshift.converter import LayoutConverter

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "bench_detector.py"


@pytest.fixture(scope="module")
def bench() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bench_detector", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_detector"] = module
    spec.loader.exec_module(module)
    return module


def test_word_list_is_well_formed(bench: ModuleType) -> None:
    items = bench.read_items()
    assert len(items) > 500
    assert {item.layout for item in items} == {"ru", "us"}
    keys = [(item.word, item.layout) for item in items]
    assert len(keys) == len(set(keys)), "a word is listed twice"
    for item in items:
        assert item.category.startswith(("ru_", "en_", "keep_")), item
        if item.category.startswith("ru_"):
            assert item.layout == "ru", item
        if item.category.startswith("en_"):
            assert item.layout == "us", item


def test_every_word_survives_the_layout_round_trip(bench: ModuleType) -> None:
    converter = LayoutConverter()
    for item in bench.read_items():
        if item.keep_only:
            continue
        other = bench.OTHER_LAYOUT[item.layout]
        wrong = converter.convert(item.word, item.layout, other)
        assert converter.convert(wrong, other, item.layout) == item.word, item


def test_both_splits_are_populated(bench: ModuleType) -> None:
    splits = {item.split for item in bench.read_items()}
    assert splits == {"dev", "test"}
