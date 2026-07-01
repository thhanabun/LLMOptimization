from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .certification_matrix import ModelCertificationTarget
from .report import make_table

LOCAL_MODEL_FIXTURE_SCHEMA_VERSION = "llm_memlab.local_model_harness.v1"


@dataclass(frozen=True)
class LocalModelFixture:
    name: str
    family: str
    model_id: str
    aliases: tuple[str, ...]
    production: bool = True


@dataclass(frozen=True)
class LocalModelMatch:
    fixture: LocalModelFixture
    path: str | None
    available: bool

    def to_target(self) -> ModelCertificationTarget:
        return ModelCertificationTarget(
            name=self.fixture.name,
            family=self.fixture.family,
            model=self.path or self.fixture.model_id,
            local_files_only=True,
            production=self.fixture.production,
        )


@dataclass(frozen=True)
class LocalModelHarnessReport:
    root: str
    matches: tuple[LocalModelMatch, ...]
    schema_version: str = LOCAL_MODEL_FIXTURE_SCHEMA_VERSION

    @property
    def available_count(self) -> int:
        return sum(1 for item in self.matches if item.available)

    def targets(self, *, available_only: bool = True) -> tuple[ModelCertificationTarget, ...]:
        return tuple(item.to_target() for item in self.matches if item.available or not available_only)

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        payload = {
            "schema_version": self.schema_version,
            "root": self.root,
            "available_count": self.available_count,
            "matches": [
                {
                    "fixture": asdict(item.fixture),
                    "path": item.path,
                    "available": item.available,
                }
                for item in self.matches
            ],
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output

    def to_text(self) -> str:
        rows = [
            (
                item.fixture.name,
                item.fixture.family,
                item.available,
                item.path or item.fixture.model_id,
            )
            for item in self.matches
        ]
        return make_table(("Fixture", "Family", "Available", "Path/model id"), rows)


DEFAULT_LOCAL_MODEL_FIXTURES: tuple[LocalModelFixture, ...] = (
    LocalModelFixture(
        "tinyllama",
        "llama",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ("tinyllama", "TinyLlama-1.1B-Chat-v1.0", "TinyLlama_TinyLlama-1.1B-Chat-v1.0"),
    ),
    LocalModelFixture(
        "qwen2.5",
        "qwen",
        "Qwen/Qwen2.5-0.5B-Instruct",
        ("qwen2.5", "qwen25", "Qwen2.5-0.5B-Instruct", "Qwen_Qwen2.5-0.5B-Instruct"),
    ),
    LocalModelFixture(
        "qwen3",
        "qwen3",
        "Qwen/Qwen3-1.7B",
        ("qwen3", "Qwen3-1.7B", "Qwen_Qwen3-1.7B"),
    ),
    LocalModelFixture(
        "mistral",
        "mistral",
        "mistralai/Mistral-7B-Instruct-v0.3",
        ("mistral", "Mistral-7B-Instruct-v0.3", "mistralai_Mistral-7B-Instruct-v0.3"),
    ),
    LocalModelFixture(
        "gemma",
        "gemma",
        "google/gemma-2-2b-it",
        ("gemma", "gemma-2-2b-it", "google_gemma-2-2b-it"),
    ),
    LocalModelFixture(
        "gemma4",
        "gemma",
        "google/gemma-4-E4B-it",
        ("gemma4", "gemma-4-E4B-it", "google_gemma-4-E4B-it"),
    ),
    LocalModelFixture(
        "phi",
        "phi",
        "microsoft/Phi-3-mini-4k-instruct",
        ("phi", "phi3", "Phi-3-mini-4k-instruct", "microsoft_Phi-3-mini-4k-instruct"),
    ),
)


def scan_local_model_fixtures(
    root: str | Path,
    fixtures: tuple[LocalModelFixture, ...] = DEFAULT_LOCAL_MODEL_FIXTURES,
) -> LocalModelHarnessReport:
    base = Path(root)
    matches = []
    for fixture in fixtures:
        path = _find_fixture_path(base, fixture)
        matches.append(LocalModelMatch(fixture, path, path is not None))
    return LocalModelHarnessReport(str(base), tuple(matches))


def _find_fixture_path(root: Path, fixture: LocalModelFixture) -> str | None:
    candidates = [root / alias for alias in fixture.aliases]
    candidates.extend(root / alias.replace("/", "_") for alias in (fixture.model_id, fixture.model_id.split("/")[-1]))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None
