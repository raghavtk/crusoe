#!/usr/bin/env python3
"""
Crusoe Pipeline CLI
===================

Entry point for running the multi-agent literature review pipeline.

Usage
-----
  python scripts/run_pipeline.py --topic "authentication tokens in web security"
  python scripts/run_pipeline.py --topic "..." --provider cerebras
  python scripts/run_pipeline.py --resume
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure the project root is on the Python path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.rule import Rule
from rich.text import Text

load_dotenv()

console = Console()


def setup_logging(log_level: str) -> None:
    """Configure loguru to write to a log file, with console at WARNING+."""
    logger.remove()
    logger.add(
        "data/crusoe.log",
        level=log_level.upper(),
        rotation="10 MB",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
    )
    # Only show WARNING and above on stderr to keep the CLI clean
    logger.add(sys.stderr, level="WARNING", format="{level}: {message}")


def load_config(path: str = "config.yaml") -> dict:
    """Load and return the config.yaml as a dict."""
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] config.yaml not found at {p.resolve()}")
        sys.exit(1)
    return yaml.safe_load(p.read_text()) or {}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="crusoe",
        description="Crusoe — Multi-Agent Literature Review Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --topic "authentication tokens in web security"
  python scripts/run_pipeline.py --topic "..." --provider cerebras
  python scripts/run_pipeline.py --resume
        """,
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Research topic to review (required unless --resume is set)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "cerebras"],
        default=None,
        help="Override the LLM provider from config.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last checkpoint instead of starting fresh",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    return parser.parse_args()


def print_banner() -> None:
    """Print the Crusoe ASCII banner."""
    banner = Text()
    banner.append("  ██████ ██████  ██    ██ ███████  ██████  ███████ \n", style="bold cyan")
    banner.append(" ██      ██   ██ ██    ██ ██      ██    ██ ██      \n", style="bold cyan")
    banner.append(" ██      ██████  ██    ██ ███████ ██    ██ █████   \n", style="bold cyan")
    banner.append(" ██      ██   ██ ██    ██      ██ ██    ██ ██      \n", style="bold cyan")
    banner.append("  ██████ ██   ██  ██████  ███████  ██████  ███████ \n", style="bold cyan")
    banner.append("\n  Multi-Agent Literature Review Pipeline", style="italic dim")
    console.print(Panel(banner, border_style="cyan", padding=(0, 2)))


def main() -> None:
    """Main entry point."""
    args = parse_args()
    config = load_config(args.config)

    setup_logging(config.get("pipeline", {}).get("log_level", "INFO"))

    print_banner()

    # Validate arguments
    if not args.resume and not args.topic:
        console.print(
            "[red]Error:[/red] --topic is required when not using --resume.\n"
            "Example: python scripts/run_pipeline.py --topic \"neural network security\""
        )
        sys.exit(1)

    # Override provider if specified
    if args.provider:
        config["llm"]["provider"] = args.provider
        console.print(f"[dim]Provider override: [bold]{args.provider}[/bold][/dim]")

    provider_name = config["llm"]["provider"]
    topic = args.topic

    console.print()
    if args.resume:
        console.print(f"[bold]Mode:[/bold] Resume from checkpoint")
    else:
        console.print(f"[bold]Topic:[/bold] {topic}")
    console.print(f"[bold]Provider:[/bold] {provider_name}")
    console.print()

    # Initialise the LLM provider
    from src.llm.providers import get_provider
    try:
        provider = get_provider(config["llm"])
    except EnvironmentError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("Set the required API key in your .env file.")
        sys.exit(1)

    # Run the pipeline with progress display
    console.print(Rule("[bold cyan]Pipeline Starting[/bold cyan]"))
    console.print()

    from src.agents.orchestrator import run_pipeline
    from src.core.state import PipelineState

    try:
        state = _run_with_progress(
            topic=topic,
            provider=provider,
            config=config,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user.[/yellow]")
        console.print(f"Progress saved to: {config['pipeline']['checkpoint_path']}")
        sys.exit(0)
    except Exception as exc:
        console.print(f"\n[red bold]Pipeline failed:[/red bold] {exc}")
        logger.exception("Pipeline error")
        sys.exit(1)

    # ── Final summary ─────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold green]Pipeline Complete[/bold green]"))
    console.print()

    _print_summary(state, config)


def _run_with_progress(
    topic: str,
    provider: Any,
    config: dict,
    resume: bool,
) -> "PipelineState":
    """
    Run the pipeline, intercepting loguru INFO messages to display
    stage completion lines in the CLI.
    """
    from src.agents.orchestrator import run_pipeline

    # We use a simple approach: run the pipeline and let loguru handle
    # internal logging, while we show a spinner + stage markers.
    stages = [
        "Topic Decomposition",
        "Discovery",
        "Enrichment",
        "Synthesis",
        "Google Sheets",
    ]

    # Intercept orchestrator log messages to print progress markers
    completed_stages: list[str] = []

    def _log_sink(message: "Any") -> None:
        record = message.record
        text: str = record["message"]
        if "✓ Topic Decomposition" in text:
            _print_stage_done(text)
        elif "✓ Discovery" in text:
            _print_stage_done(text)
        elif "✓ Enrichment" in text:
            _print_stage_done(text)
        elif "✓ Synthesis" in text:
            _print_stage_done(text)
        elif "✓ Google Sheets" in text:
            _print_stage_done(text)

    sink_id = logger.add(_log_sink, level="INFO", format="{message}")

    try:
        state = run_pipeline(
            topic=topic,
            provider=provider,
            config=config,
            resume=resume,
        )
    finally:
        logger.remove(sink_id)

    return state


def _print_stage_done(message: str) -> None:
    """Print a green checkmark line for a completed stage."""
    # Extract the part after [Orchestrator]
    parts = message.split("] ", 1)
    display = parts[1] if len(parts) > 1 else message
    console.print(f"  [green]✓[/green] {display}")


def _print_summary(state: "PipelineState", config: dict) -> None:
    """Print the final summary table."""
    console.print(f"  [green]✓[/green] [bold]Topic:[/bold] {state.topic}")
    console.print(f"  [green]✓[/green] [bold]Clusters:[/bold] {len(state.keyword_clusters)}")
    console.print(f"  [green]✓[/green] [bold]Papers found:[/bold] {len(state.papers_raw)}")
    console.print(f"  [green]✓[/green] [bold]Papers enriched:[/bold] {len(state.papers_enriched)}")

    priority = sum(1 for p in state.papers_enriched if p.get("priority_read"))
    console.print(f"  [green]✓[/green] [bold]Priority reads:[/bold] {priority}")

    themes = state.synthesis.get("key_themes", [])
    console.print(f"  [green]✓[/green] [bold]Key themes identified:[/bold] {len(themes)}")

    if state.sheet_url:
        console.print(f"  [green]✓[/green] [bold]Google Sheet:[/bold] [link={state.sheet_url}]{state.sheet_url}[/link]")
    else:
        console.print("  [yellow]⚠[/yellow] Google Sheets write was skipped or failed.")

    if state.errors:
        console.print()
        console.print(f"  [yellow]⚠ {len(state.errors)} non-fatal error(s) occurred.[/yellow]")
        for err in state.errors[:3]:
            console.print(f"    - {err}")
        if len(state.errors) > 3:
            console.print(f"    … and {len(state.errors) - 3} more (see data/crusoe.log)")

    console.print()
    checkpoint = config["pipeline"]["checkpoint_path"]
    console.print(f"[dim]Checkpoint saved: {checkpoint}[/dim]")
    console.print()


if __name__ == "__main__":
    main()
