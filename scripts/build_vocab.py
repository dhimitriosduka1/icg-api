"""Rebuild artifacts/vocab.json from the caption corpus.

The token order must match the one the model was trained with, so only rerun
this when retraining the model on a new corpus.

    uv run scripts/build_vocab.py [--corpus corpus/annotations.csv] [--output artifacts/vocab.json]
"""

import argparse
import csv
from pathlib import Path

from icg_api.vocabulary import Vocabulary


def read_captions(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as file:
        return [row["caption"] for row in csv.DictReader(file, delimiter="|")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("corpus/annotations.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/vocab.json"))
    parser.add_argument("--frequency-threshold", type=int, default=5)
    args = parser.parse_args()

    vocabulary = Vocabulary.build(read_captions(args.corpus), args.frequency_threshold)
    vocabulary.save(args.output)
    print(f"Wrote {len(vocabulary)} tokens to {args.output}")


if __name__ == "__main__":
    main()
