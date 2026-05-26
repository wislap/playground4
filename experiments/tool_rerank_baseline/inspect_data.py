from __future__ import annotations

import argparse
import sys
import tomllib
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from data import load_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    bundle = load_dataset(config)
    examples = bundle.train_examples + bundle.val_examples
    labels = [example.label for example in examples]
    group_sizes = [len(group.examples) for group in bundle.train_groups + bundle.val_groups]

    print("groups", len(bundle.train_groups) + len(bundle.val_groups))
    print("train_groups", len(bundle.train_groups), "val_groups", len(bundle.val_groups))
    print("pairs", len(examples), "train_pairs", len(bundle.train_examples), "val_pairs", len(bundle.val_examples))
    print("candidate_size min/max/avg", min(group_sizes), max(group_sizes), round(sum(group_sizes) / len(group_sizes), 2))
    print("label min/max/avg", round(min(labels), 4), round(max(labels), 4), round(sum(labels) / len(labels), 4))
    print("label >=0.8", sum(label >= 0.8 for label in labels))
    print("scenario", Counter(example.scenario_type for example in examples))
    print("relevance_mode", Counter(example.relevance_mode for example in examples))


if __name__ == "__main__":
    main()
