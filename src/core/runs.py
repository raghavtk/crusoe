"""Resolve run-local storage before request planning or provider initialization."""

from copy import deepcopy
from pathlib import Path
from uuid import uuid4


def configure_run(config: dict, config_path: str | Path, *, resume: bool = False,
                  run_dir: str | Path | None = None) -> dict:
    result = deepcopy(config)
    pipeline = result.setdefault("pipeline", {})
    legacy = run_dir is None and bool(pipeline.get("checkpoint_path"))
    if legacy:
        checkpoint = Path(pipeline["checkpoint_path"]).resolve()
    else:
        if run_dir is None:
            if resume:
                raise ValueError("Resume requires --run-dir for the research run to continue")
            root = Path(pipeline.get("runs_directory", "data/runs"))
            if not root.is_absolute():
                root = Path(config_path).resolve().parent / root
            run_dir = root / uuid4().hex
        directory = Path(run_dir).resolve()
        checkpoint = directory / "checkpoint.json"
        if not resume and directory.exists() and any(directory.iterdir()):
            raise ValueError("Run directory is not empty; use --resume or choose a new directory")
        pipeline["run_dir"] = str(directory)
        pipeline["log_file"] = str(directory / "crusoe.log")
        result.setdefault("output", {}).setdefault("xlsx", {})["directory"] = str(directory / "outputs")
    if resume and not checkpoint.is_file():
        raise ValueError("The requested checkpoint does not exist; no new run was started")
    if not resume and checkpoint.exists():
        raise ValueError("Checkpoint already exists; use --resume or choose a new run directory")
    pipeline["checkpoint_path"] = str(checkpoint)
    return result
