"""Dynamic Skill discovery, matching, and prompt rendering.

Each skill is a directory containing a ``SKILL.md`` document with a YAML
front matter block. Business operators can update a Skill without changing
Python code; ``SkillManager.reload()`` atomically replaces the active catalog.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import yaml

from config.settings import settings

logger = logging.getLogger(__name__)

_FRONT_MATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    """A validated, runtime-loadable operating procedure."""

    name: str
    description: str
    agents: tuple[str, ...]
    keywords: tuple[str, ...]
    content: str
    source_path: str

    def matches(self, agent_type: str, message: str) -> bool:
        """Return whether this skill applies to an agent/message pair."""
        normalized_agent = agent_type.strip().lower()
        eligible_agent = "*" in self.agents or normalized_agent in self.agents
        if not eligible_agent:
            return False
        if not self.keywords:
            return True
        normalized_message = message.lower()
        return any(keyword in normalized_message for keyword in self.keywords)


class SkillLoadError(ValueError):
    """Raised when a SKILL.md document does not meet the loader contract."""


class SkillManager:
    """Thread-safe catalog of Skills with explicit hot-reload support."""

    def __init__(self, skills_dir: str | Path | None = None) -> None:
        configured_path = skills_dir or settings.skills_dir
        self._skills_dir = Path(configured_path)
        if not self._skills_dir.is_absolute():
            self._skills_dir = settings.project_root / self._skills_dir
        self._lock = RLock()
        self._skills: tuple[Skill, ...] = ()
        self.reload()

    @property
    def skills_dir(self) -> Path:
        return self._skills_dir

    def reload(self) -> list[Skill]:
        """Parse every SKILL.md and atomically publish the valid catalog.

        A malformed document is logged and skipped. Existing active Skills stay
        usable if every document in a reload happens to be malformed.
        """
        discovered: list[Skill] = []
        errors: list[str] = []
        if not self._skills_dir.exists():
            logger.warning("Skills directory does not exist: %s", self._skills_dir)
            with self._lock:
                self._skills = ()
            return []

        for skill_path in sorted(self._skills_dir.glob("*/SKILL.md")):
            try:
                discovered.append(self._load_file(skill_path))
            except SkillLoadError as exc:
                errors.append(f"{skill_path}: {exc}")
                logger.warning("Skipping invalid skill: %s", errors[-1])

        names = [skill.name for skill in discovered]
        if len(names) != len(set(names)):
            raise SkillLoadError("Skill names must be unique")

        with self._lock:
            self._skills = tuple(discovered)
        logger.info("Loaded %d skills from %s", len(discovered), self._skills_dir)
        return list(discovered)

    def list_skills(self) -> list[dict[str, Any]]:
        """Return metadata intended for an administrative API response."""
        with self._lock:
            return [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "agents": list(skill.agents),
                    "keywords": list(skill.keywords),
                    "source_path": skill.source_path,
                }
                for skill in self._skills
            ]

    def match(self, agent_type: str, message: str) -> list[Skill]:
        """Select Skills whose agent and keyword conditions both match."""
        with self._lock:
            return [skill for skill in self._skills if skill.matches(agent_type, message)]

    def render_prompt(self, agent_type: str, message: str) -> str:
        """Render matching operational rules into a bounded prompt section."""
        matched = self.match(agent_type, message)
        if not matched:
            return ""
        sections = [f"## Operational skill: {skill.name}\n{skill.content.strip()}" for skill in matched]
        rendered = "\n\n".join(sections)
        if len(rendered) > settings.skills_max_prompt_chars:
            logger.warning("Matched skill prompt truncated from %d characters", len(rendered))
            return rendered[: settings.skills_max_prompt_chars] + "\n[Skill prompt truncated]"
        return rendered

    @staticmethod
    def _load_file(path: Path) -> Skill:
        raw = path.read_text(encoding="utf-8")
        match = _FRONT_MATTER_PATTERN.match(raw)
        if not match:
            raise SkillLoadError("Expected YAML front matter delimited by ---")
        try:
            metadata = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            raise SkillLoadError(f"Invalid YAML: {exc}") from exc
        if not isinstance(metadata, dict):
            raise SkillLoadError("YAML front matter must be a mapping")

        name = SkillManager._required_string(metadata, "name")
        description = SkillManager._required_string(metadata, "description")
        agents = SkillManager._string_sequence(metadata, "agents")
        keywords = SkillManager._string_sequence(metadata, "keywords", allow_empty=True)
        content = match.group(2).strip()
        if not content:
            raise SkillLoadError("Skill body cannot be empty")
        return Skill(
            name=name,
            description=description,
            agents=tuple(item.lower() for item in agents),
            keywords=tuple(item.lower() for item in keywords),
            content=content,
            source_path=str(path),
        )

    @staticmethod
    def _required_string(metadata: dict[str, Any], key: str) -> str:
        value = metadata.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SkillLoadError(f"'{key}' must be a non-empty string")
        return value.strip()

    @staticmethod
    def _string_sequence(
        metadata: dict[str, Any], key: str, *, allow_empty: bool = False
    ) -> list[str]:
        value = metadata.get(key)
        if not isinstance(value, list) or (not allow_empty and not value):
            requirement = "a list" if allow_empty else "a non-empty list"
            raise SkillLoadError(f"'{key}' must be {requirement}")
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise SkillLoadError(f"'{key}' entries must be non-empty strings")
        return [item.strip() for item in value]
