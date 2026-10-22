"""Inspect synthesis claims beside cited abstracts and save human judgments offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError

from src.core.state import PipelineState
from src.evaluation.claim_review import (
    build_snapshot, load_review, new_review, record_judgment, review_counts, review_lock, save_review,
)


def display(text: str) -> str:
    """Render source data literally without executing terminal control sequences."""
    return "".join(character if character.isprintable() or character in "\n\t"
                   else f"\\u{ord(character):04x}" for character in text)


def print_counts(review) -> None:
    print(f"{len(review.judgments)}/{len(review.snapshot.claims)} reviewed")
    print(" / ".join(f"{count} {status}" for status, count in review_counts(review).items()))


def main() -> int:
    # Pipes and older Windows terminals may not support every source character.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="Separate review JSON; defaults beside checkpoint")
    parser.add_argument("--reviewer", default="", help="Reviewer name recorded with each decision")
    parser.add_argument("--revisit", action="store_true", help="Include already reviewed claims")
    parser.add_argument("--summary", action="store_true", help="Show judgment counts without prompting")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = (args.output or checkpoint.with_name(checkpoint.stem + ".claim-review.json")).resolve()
    if output == checkpoint or (output.exists() and checkpoint.exists() and output.samefile(checkpoint)):
        parser.error("Review output must be separate from the pipeline checkpoint")

    review = None
    try:
        snapshot = build_snapshot(PipelineState.load(checkpoint))
        with review_lock(output):
            review = load_review(output, snapshot) if output.exists() else new_review(snapshot)
            print(display(f"Topic: {snapshot.topic}"))
            print("Cited context is the full stored abstract, not a claim-aligned quotation.")
            print("Synthesis may have seen a normalized excerpt of at most 1,200 characters.")
            if snapshot.uncited_sections:
                print("Not claim-aligned: " + ", ".join(snapshot.uncited_sections))
            print_counts(review)
            if args.summary:
                return 0
            for index, claim in enumerate(snapshot.claims, 1):
                if claim.claim_id in review.judgments and not args.revisit:
                    continue
                print(display(f"\nClaim {index}/{len(snapshot.claims)} [{claim.path}]\n{claim.label}\n{claim.text}"))
                if claim.model_confidence is not None:
                    print(f"Model confidence: {claim.model_confidence} (not a human judgment)")
                if claim.claim_id in review.judgments:
                    print("Previous judgment: " + review.judgments[claim.claim_id].status)
                for source in claim.evidence:
                    print(display(f"\nCited paper {source.paper_id}: {source.title}"))
                    print("Source record missing." if source.missing else
                          display(source.abstract) if source.abstract else "No abstract available.")
                while True:
                    choice = input("\n1 supported / 2 unsupported / 3 uncertain / s skip / q quit: ").strip().lower()
                    if choice in {"1", "2", "3", "s", "q"}:
                        break
                    print("Enter 1, 2, 3, s, or q.")
                if choice == "q":
                    break
                if choice == "s":
                    continue
                note = input("Optional note: ")
                updated = record_judgment(review, claim.claim_id,
                                          {"1": "supported", "2": "unsupported", "3": "uncertain"}[choice],
                                          note=note, reviewer=args.reviewer)
                save_review(updated, output)
                review = updated
            print_counts(review)
            print(display(f"Review output: {output}"))
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\nStopped. Previously saved judgments are preserved.")
        if review is not None:
            print_counts(review)
        return 0
    except ValidationError:
        print("Review failed: invalid checkpoint or review artifact schema.", file=sys.stderr)
    except ValueError as exc:
        print("Review failed: " + display(str(exc)), file=sys.stderr)
    except OSError as exc:
        print("Review storage failed: " + type(exc).__name__, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
