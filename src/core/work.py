"""Strict, content-addressed manifests for reusable discovery and curator units."""

from hashlib import sha256
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


class UnitResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    result: list[dict]
    checksum: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class StageWork(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal[1]
    kind: Literal["discovery", "curator"]
    fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    plan: list[list[StrictStr]]
    completed: dict[StrictStr, UnitResult] = Field(default_factory=dict)
    output_checksum: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("version", mode="before")
    @classmethod
    def _version(cls, value):
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported work version")
        return value

    @property
    def complete(self) -> bool:
        return len(self.completed) == len(self.plan)


def load_work(raw: dict, *, kind: str, plan: list[list[str]], inputs: dict) -> StageWork:
    expected = StageWork(version=1, kind=kind, plan=plan,
                         fingerprint=fingerprint({"version": 1, "kind": kind, "plan": plan, **inputs}))
    if not raw:
        return expected
    try:
        existing = StageWork.model_validate(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid {kind} work artifact") from exc
    if existing.kind != kind:
        raise ValueError(f"Invalid {kind} work kind")
    for key, unit in existing.completed.items():
        if key not in {str(index) for index in range(len(existing.plan))}:
            raise ValueError(f"Invalid {kind} work unit index")
        if fingerprint(unit.result) != unit.checksum:
            raise ValueError(f"Invalid {kind} work checksum")
    if existing.fingerprint != expected.fingerprint:
        return expected
    if existing.plan != expected.plan:
        raise ValueError(f"Invalid {kind} work plan")
    return existing


def store_unit(work: StageWork, index: int, result: list[dict]) -> None:
    work.completed[str(index)] = UnitResult(result=result, checksum=fingerprint(result))


def provider_identity(provider) -> str:
    seen = set()
    while hasattr(provider, "delegate") and id(provider) not in seen:
        seen.add(id(provider))
        provider = provider.delegate
    return f"{type(provider).__module__}.{type(provider).__qualname__}:" + repr({
        "model": getattr(provider, "_model_name", None) or getattr(provider, "_model", None),
        "temperature": getattr(provider, "_temperature", None),
    })
