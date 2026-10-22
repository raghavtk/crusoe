"""Score saved, manually judged discovery comparisons without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.discovery import compare_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSON array of saved comparison cases")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        parser.error("cases must be a nonempty JSON array")
    print(json.dumps([compare_retrieval(case, top_k=args.top_k) for case in cases], indent=2))


if __name__ == "__main__":
    main()
