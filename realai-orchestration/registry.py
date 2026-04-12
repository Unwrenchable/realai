"""Agent registry for RealAI orchestration.

Loads agent definitions from ``.agentx/agents.json`` in the repository root
(walking up from the current working directory to find the file), falling back
to the package-bundled ``.agentx/agents.json`` when not found.

The registry format matches the `agent-tools <https://github.com/Unwrenchable/agent-tools>`_
schema so agent definitions can be shared across repos.

Example::

    from realai_orchestration.registry import AgentRegistry

    registry = AgentRegistry()
    all_agents = registry.list_agents()
    architect = registry.get("realai-architect")
    realai_team = registry.find("realai")   # agents with "realai" in id/tags
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AgentDefinition:
    """Typed representation of an agent entry in agents.json.

    Attributes:
        id: Unique slug, e.g. ``"realai-capability-engineer"``.
        role: Short human-readable role title.
        description: Full system-prompt-style description of the agent's
            responsibilities.  Used as the ``role`` argument to
            :class:`~agent.BaseAgent`.
        tags: Free-form labels for searching and grouping.
        capabilities: Named capability identifiers the agent possesses.
        required_tools: Tools the agent needs to function correctly.
        preferred_profile: One of ``safe``, ``balanced``, ``power``.
        risk_level: Qualitative risk rating (``low``, ``medium``, ``high``).
    """

    id: str
    role: str
    description: str
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    required_tools: List[str] = field(default_factory=list)
    preferred_profile: str = "balanced"
    risk_level: str = "medium"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentDefinition":
        """Construct an :class:`AgentDefinition` from a raw JSON dict.

        Args:
            data: Dict loaded from agents.json.

        Returns:
            A populated :class:`AgentDefinition`.
        """
        return cls(
            id=data["id"],
            role=data.get("role", data["id"]),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            capabilities=data.get("capabilities", []),
            required_tools=data.get("required_tools", []),
            preferred_profile=data.get("preferred_profile", "balanced"),
            risk_level=data.get("risk_level", "medium"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this definition back to a plain dict."""
        return {
            "id": self.id,
            "role": self.role,
            "description": self.description,
            "tags": self.tags,
            "capabilities": self.capabilities,
            "required_tools": self.required_tools,
            "preferred_profile": self.preferred_profile,
            "risk_level": self.risk_level,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Loads and queries the agent registry from ``.agentx/agents.json``.

    Resolution order:
    1. Path passed to the constructor.
    2. Walk upward from the current working directory for a ``.agentx/agents.json``
       file, stopping at the first ``.git`` boundary.
    3. Fallback: the bundled ``.agentx/agents.json`` at the repository root
       (resolved relative to this module's location).

    Args:
        path: Optional explicit path to an ``agents.json`` file.

    Example::

        registry = AgentRegistry()
        team = registry.find("realai")
        print([a.id for a in team])
    """

    def __init__(self, path: Optional[str] = None) -> None:
        self._path: Optional[Path] = Path(path) if path else None
        self._agents: Optional[Dict[str, AgentDefinition]] = None

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_path(self) -> Optional[Path]:
        """Return the path of the agents.json file to load."""
        if self._path is not None:
            return self._path if self._path.is_file() else None

        # Walk up from CWD looking for .agentx/agents.json
        current = Path(os.getcwd())
        for directory in [current] + list(current.parents):
            candidate = directory / ".agentx" / "agents.json"
            if candidate.is_file():
                return candidate
            if (directory / ".git").is_dir():
                break

        # Fallback: bundled file relative to this module
        bundled = Path(__file__).resolve().parent.parent / ".agentx" / "agents.json"
        if bundled.is_file():
            return bundled

        return None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, AgentDefinition]:
        """Load agents from the resolved JSON file."""
        if self._agents is not None:
            return self._agents

        path = self._resolve_path()
        if path is None:
            self._agents = {}
            return self._agents

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._agents = {}
            return self._agents

        if not isinstance(raw, list):
            self._agents = {}
            return self._agents

        self._agents = {}
        for item in raw:
            if not isinstance(item, dict) or "id" not in item:
                continue
            defn = AgentDefinition.from_dict(item)
            self._agents[defn.id] = defn

        return self._agents

    def reload(self) -> None:
        """Reload the registry from disk, discarding any cached data."""
        self._agents = None

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def list_agents(self) -> List[AgentDefinition]:
        """Return all registered agents.

        Returns:
            List of :class:`AgentDefinition` instances.
        """
        return list(self._load().values())

    def get(self, agent_id: str) -> Optional[AgentDefinition]:
        """Return the agent with the given *agent_id*, or None.

        Args:
            agent_id: Agent ID to look up.

        Returns:
            :class:`AgentDefinition` or None.
        """
        return self._load().get(agent_id)

    def find(self, query: str) -> List[AgentDefinition]:
        """Search agents by id, role, tags, capabilities, or description.

        The search is case-insensitive.  An agent matches if *query* appears
        anywhere in its id, role, tags list, capabilities list, or
        description.

        Args:
            query: Search string.

        Returns:
            List of matching :class:`AgentDefinition` instances.
        """
        q = query.lower()
        results: List[AgentDefinition] = []
        for defn in self._load().values():
            if (
                q in defn.id.lower()
                or q in defn.role.lower()
                or q in defn.description.lower()
                or any(q in t.lower() for t in defn.tags)
                or any(q in c.lower() for c in defn.capabilities)
            ):
                results.append(defn)
        return results

    def filter_by_profile(self, profile: str) -> List[AgentDefinition]:
        """Return agents whose ``preferred_profile`` matches *profile*.

        Args:
            profile: One of ``"safe"``, ``"balanced"``, ``"power"``.

        Returns:
            Matching :class:`AgentDefinition` instances.
        """
        return [
            defn
            for defn in self._load().values()
            if defn.preferred_profile == profile
        ]

    def filter_by_risk(self, risk_level: str) -> List[AgentDefinition]:
        """Return agents whose ``risk_level`` matches *risk_level*.

        Args:
            risk_level: One of ``"low"``, ``"medium"``, ``"high"``.

        Returns:
            Matching :class:`AgentDefinition` instances.
        """
        return [
            defn
            for defn in self._load().values()
            if defn.risk_level == risk_level
        ]

    def __len__(self) -> int:
        return len(self._load())

    def __repr__(self) -> str:  # pragma: no cover
        return f"AgentRegistry(agents={len(self)}, path={self._resolve_path()})"
