"""Held-out benchmark for wrong-layout detection (ru <-> en).

Unlike ``scripts/measure_detector.py``, which slices the very corpora the
detector loads, this script scores a hand-written word set
(``scripts/bench_detector_words.tsv``) that is not derived from
``ngrams_*.json`` or ``dictionaries/*-50k.txt``. It reports:

* **hit rate** -- words typed on the wrong layout that the detector sends back to
  the right one (and whose conversion restores the word exactly);
* **false corrections** -- words typed correctly that the detector would still
  move to another layout. The ``keep_*`` categories (code, transliteration,
  abbreviations, slang, ru/en look-alikes) are measured for this only;
* both broken down by category, by word length and by whether the word is in the
  shipped 10k vocabulary.

The detector is built from ``Config()`` defaults exactly as the daemon builds it,
and the daemon's own internal-separator guard is applied on top, so the numbers
describe what a user with a default install gets on a Space-terminated word.
``--extended`` also loads the bundled 50k lists as extended dictionaries.

Every word carries a fixed ``dev``/``test`` tag (CRC32 of the word), so a change
can be tuned on ``dev`` and confirmed on ``test``.

Run with ``PYTHONPATH=src python scripts/bench_detector.py``; ``--json`` prints
machine-readable totals and ``--misses`` lists every miss and false correction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zlib
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from klyrshift.config import Config  # noqa: E402
from klyrshift.converter import LayoutConverter  # noqa: E402
from klyrshift.daemon import _INTERNAL_SEPARATOR_RE  # noqa: E402
from klyrshift.detector import LanguageDetector  # noqa: E402

WORDS_FILE = Path(__file__).resolve().parent / "bench_detector_words.tsv"
OTHER_LAYOUT = {"ru": "us", "us": "ru"}
LAYOUT_LANGUAGE = {"ru": "ru", "us": "en"}
LENGTH_BUCKETS = ((2, 2, "2"), (3, 3, "3"), (4, 4, "4"), (5, 6, "5-6"), (7, 9, "7-9"))


@dataclass(frozen=True)
class Item:
    """One benchmark word."""

    category: str
    layout: str
    word: str

    @property
    def split(self) -> str:
        return "dev" if zlib.crc32(self.word.encode("utf-8")) % 2 == 0 else "test"

    @property
    def keep_only(self) -> bool:
        return self.category.startswith("keep_")


def read_items(path: Path = WORDS_FILE) -> list[Item]:
    items: list[Item] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        category, layout, word = line.split("\t")
        items.append(Item(category, layout, word))
    return items


def length_bucket(word: str) -> str:
    for low, high, name in LENGTH_BUCKETS:
        if low <= len(word) <= high:
            return name
    return "10+" if len(word) >= 10 else "1"


def build_detector(extended_dir: str = "") -> LanguageDetector:
    """Build the detector the way ``KlyrshiftDaemon`` does from ``Config()``."""
    config = Config()
    converter = LayoutConverter()
    return LanguageDetector(
        converter=converter,
        stop_words=config.stop_words,
        min_word_length=config.min_word_length,
        confidence_threshold=config.confidence_threshold,
        languages=tuple(config.languages),
        plausibility_check=config.plausibility_check,
        structural_boundaries=config.structural_boundaries,
        identifier_guard=config.identifier_guard,
        plausibility_floor=config.plausibility_floor,
        max_consecutive_consonants=config.max_consecutive_consonants,
        min_vowel_ratio=config.min_vowel_ratio,
        extended_dictionary_dir=extended_dir,
        thematic_categories=config.installed_dict_categories,
    )


def daemon_target(detector: LanguageDetector, text: str, layout: str) -> str | None:
    """Return the layout the daemon would convert ``text`` to, or ``None``."""
    target = detector.target_layout(text, layout)
    if target is None:
        return None
    if Config().ignore_emails_urls and _INTERNAL_SEPARATOR_RE.search(text):
        return None
    return target


class Tally:
    """Hits and false corrections for one slice of the word set."""

    def __init__(self) -> None:
        self.wrong = 0
        self.hit = 0
        self.correct = 0
        self.false = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "wrong": self.wrong,
            "hit": self.hit,
            "hit_rate": round(100.0 * self.hit / self.wrong, 1) if self.wrong else 0.0,
            "correct": self.correct,
            "false": self.false,
            "false_rate": round(100.0 * self.false / self.correct, 1) if self.correct else 0.0,
        }


def run(
    detector: LanguageDetector, items: list[Item], split: str
) -> tuple[dict[str, dict[str, Tally]], list[str]]:
    converter = detector.converter
    vocabulary = {lang: detector.ordered_vocabulary(lang) for lang in ("ru", "en")}
    known = {lang: set(words) for lang, words in vocabulary.items()}
    tables: dict[str, dict[str, Tally]] = defaultdict(lambda: defaultdict(Tally))
    notes: list[str] = []
    for item in items:
        if split != "all" and item.split != split:
            continue
        language = LAYOUT_LANGUAGE[item.layout]
        in_vocab = "in-vocab" if item.word.lower() in known[language] else "out-of-vocab"
        keys = [("total", "all")]
        keys.append(("category", item.category))
        if not item.keep_only:
            keys += [("length", length_bucket(item.word)), ("vocabulary", in_vocab)]
            keys.append(("direction", f"{language} typed on {OTHER_LAYOUT[item.layout]}"))
        tallies = [tables[table][row] for table, row in keys]

        moved = daemon_target(detector, item.word, item.layout)
        for tally in tallies:
            tally.correct += 1
            tally.false += moved is not None
        if moved is not None:
            notes.append(f"FALSE  {item.category:<14} {item.word!r} ({item.layout} -> {moved})")
        if item.keep_only:
            continue

        other = OTHER_LAYOUT[item.layout]
        wrong = converter.convert(item.word, item.layout, other)
        target = daemon_target(detector, wrong, other)
        fixed = target == item.layout and converter.convert(wrong, other, item.layout) == item.word
        for tally in tallies:
            tally.wrong += 1
            tally.hit += fixed
        if not fixed:
            notes.append(f"MISS   {item.category:<14} {item.word!r} typed as {wrong!r}")
    return tables, notes


TABLE_ORDER = ("total", "direction", "category", "length", "vocabulary")


def render(tables: dict[str, dict[str, Tally]]) -> Iterator[str]:
    header = f"{'':<22}{'hit':>12}{'rate':>8}{'false':>12}{'rate':>8}"
    for table in TABLE_ORDER:
        rows = tables.get(table)
        if not rows:
            continue
        yield ""
        yield f"by {table}" if table != "total" else "total"
        yield header
        yield "-" * len(header)
        for name in sorted(rows, key=_row_order):
            data = rows[name].as_dict()
            hit = f"{data['hit']}/{data['wrong']}" if data["wrong"] else "-"
            hit_rate = f"{data['hit_rate']:.1f}%" if data["wrong"] else "-"
            false = f"{data['false']}/{data['correct']}"
            false_rate = f"{data['false_rate']:.1f}%"
            yield f"{name:<22}{hit:>12}{hit_rate:>8}{false:>12}{false_rate:>8}"


def _row_order(name: str) -> tuple[int, str]:
    for index, (_, _, bucket) in enumerate(LENGTH_BUCKETS):
        if name == bucket:
            return index, name
    return (len(LENGTH_BUCKETS), name) if name == "10+" else (len(LENGTH_BUCKETS) + 1, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--extended", action="store_true", help="also load the bundled 50k lists")
    parser.add_argument("--json", action="store_true", help="print totals as JSON")
    parser.add_argument("--misses", action="store_true", help="list misses and false fixes")
    args = parser.parse_args()

    items = read_items()
    with tempfile.TemporaryDirectory(prefix="klyrshift-bench-") as tmp:
        extended_dir = ""
        if args.extended:
            for language in ("ru", "en"):
                source = REPO_ROOT / "dictionaries" / f"{language}-50k.txt"
                shutil.copyfile(source, Path(tmp) / f"{language}.txt")
            extended_dir = tmp
        detector = build_detector(extended_dir)
        tables, notes = run(detector, items, args.split)

    if args.json:
        payload = {
            table: {row: tally.as_dict() for row, tally in rows.items()}
            for table, rows in tables.items()
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        mode = "extended 50k" if args.extended else "default 10k"
        print(f"words: {args.split} split, vocabulary: {mode}")
        for line in render(tables):
            print(line)
    if args.misses:
        print()
        print("\n".join(notes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
