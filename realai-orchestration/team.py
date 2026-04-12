"""Pre-built RealAI agent team ("hive mind").

:class:`RealAITeam` is a convenience factory that reads the agent registry,
instantiates :class:`~agent.BaseAgent` objects for every definition (or a
specified subset), wires them all to a single :class:`~memory.SharedMemory`
instance, and returns a ready-to-use :class:`~orchestrator.Orchestrator`.

The shared memory acts as the "hive mind": every agent reads the same
accumulated context and writes its results back into it, so later agents
always have access to what earlier agents produced.

Example::

    from realai import RealAIClient
    from realai_orchestration.team import RealAITeam

    client = RealAIClient()
    orch = RealAITeam.build(client)

    # Auto-route a task to the best-matched agent
    result = orch.route("Design the architecture for a new embeddings endpoint")
    print(result["output"])

    # Run the full hive mind in sequence
    result = orch.run_pipeline("Build and document a new code-execution capability")
    print(result["final_output"])

    # Execute a JSON workflow file
    result = RealAITeam.run_workflow(client, "realai-orchestration/workflows/build-capability.json")
    print(result["final_output"])
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent import BaseAgent
from .memory import SharedMemory
from .orchestrator import Orchestrator
from .registry import AgentDefinition, AgentRegistry


class RealAITeam:
    """Factory and namespace for the RealAI specialist hive-mind team.

    All methods are class-methods so you never need to instantiate this class.
    """

    # Default ordered pipeline for the full hive-mind build workflow.
    # These IDs match the agents defined in .agentx/agents.json.
    DEFAULT_PIPELINE: List[str] = [
        "realai-repo-architect",
        "realai-capability-dev",
        "realai-fullstack-dev",
        "realai-provider-specialist",
        "realai-security-hardener",
        "realai-qa-pilot",
        "realai-documentation-pilot",
        "realai-implementation-pilot",
        "realai-orchestrator",
    ]

    @classmethod
    def build(
        cls,
        realai_client: Any,
        agent_ids: Optional[List[str]] = None,
        registry: Optional[AgentRegistry] = None,
        memory: Optional[SharedMemory] = None,
    ) -> Orchestrator:
        """Build a hive-mind :class:`~orchestrator.Orchestrator`.

        Loads agent definitions from the registry, creates
        :class:`~agent.BaseAgent` instances for each definition, attaches
        them all to a single shared :class:`~memory.SharedMemory`, and
        returns a wired :class:`~orchestrator.Orchestrator`.

        Args:
            realai_client: A :class:`~realai.RealAIClient` instance (or any
                object with a compatible ``chat.completions.create`` method).
            agent_ids: Optional list of agent IDs to include.  If omitted,
                *all* agents in the registry are included.
            registry: Optional pre-loaded :class:`~registry.AgentRegistry`.
                A new default instance is created when not provided.
            memory: Optional :class:`~memory.SharedMemory` to use as the
                hive mind.  A fresh instance is created when not provided.

        Returns:
            A fully wired :class:`~orchestrator.Orchestrator`.

        Raises:
            ValueError: If *agent_ids* is provided but none of the specified
                IDs exist in the registry.
        """
        reg = registry or AgentRegistry()
        mem = memory or SharedMemory()
        orch = Orchestrator(realai_client=realai_client, memory=mem)

        if agent_ids is not None:
            definitions: List[AgentDefinition] = []
            for aid in agent_ids:
                defn = reg.get(aid)
                if defn is not None:
                    definitions.append(defn)
            if not definitions:
                raise ValueError(
                    f"None of the requested agent IDs {agent_ids!r} "
                    "were found in the registry."
                )
        else:
            definitions = reg.list_agents()

        for defn in definitions:
            agent = BaseAgent(
                name=defn.id,
                role=defn.description or defn.role,
                realai_client=realai_client,
            )
            try:
                orch.add_agent(agent)
            except ValueError:
                # Agent already registered — skip duplicates silently
                pass

        return orch

    @classmethod
    def build_realai_team(
        cls,
        realai_client: Any,
        memory: Optional[SharedMemory] = None,
    ) -> Orchestrator:
        """Build the core RealAI specialist team (8 agents).

        Attempts to load agents in :attr:`DEFAULT_PIPELINE` order from the
        registry.  Any agent IDs not found in the registry are skipped
        gracefully rather than raising.

        Args:
            realai_client: RealAI client instance.
            memory: Optional shared memory for the hive mind.

        Returns:
            :class:`~orchestrator.Orchestrator` pre-populated with the
            RealAI specialist team.
        """
        reg = AgentRegistry()
        mem = memory or SharedMemory()
        orch = Orchestrator(realai_client=realai_client, memory=mem)

        for aid in cls.DEFAULT_PIPELINE:
            defn = reg.get(aid)
            if defn is None:
                # Fallback: create a minimal agent even without a registry entry
                defn = AgentDefinition(
                    id=aid,
                    role=aid.replace("-", " ").title(),
                    description=(
                        f"You are the {aid.replace('-', ' ')} for the RealAI project. "
                        "You help make RealAI the AI tool every other AI wants to be."
                    ),
                )
            agent = BaseAgent(
                name=defn.id,
                role=defn.description or defn.role,
                realai_client=realai_client,
            )
            try:
                orch.add_agent(agent)
            except ValueError:
                pass

        return orch

    @classmethod
    def run_workflow(
        cls,
        realai_client: Any,
        workflow_path: str,
        memory: Optional[SharedMemory] = None,
    ) -> Dict[str, Any]:
        """Load and execute a JSON workflow file.

        The workflow JSON must be a dict with a ``"steps"`` array.  Each step
        must have an ``"agent_id"`` and a ``"task"`` key.

        Args:
            realai_client: RealAI client instance.
            workflow_path: Path to the workflow JSON file.
            memory: Optional shared memory for the hive mind.

        Returns:
            Dict with ``final_output``, ``steps``, ``memory``, and
            ``success`` keys (same as
            :meth:`~orchestrator.Orchestrator.run_pipeline`).

        Raises:
            FileNotFoundError: If *workflow_path* does not exist.
            ValueError: If the JSON is malformed or missing required keys.
        """
        path = Path(workflow_path)
        if not path.is_file():
            raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

        try:
            workflow = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in workflow file: {exc}") from exc

        steps = workflow.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(
                "Workflow JSON must contain a non-empty 'steps' array."
            )

        # Collect agent IDs referenced by the workflow
        agent_ids: List[str] = []
        tasks: Dict[str, str] = {}
        for step in steps:
            if not isinstance(step, dict):
                continue
            aid = step.get("agent_id", "")
            task = step.get("task", "")
            if aid and aid not in agent_ids:
                agent_ids.append(aid)
            if aid:
                tasks[aid] = task

        orch = cls.build(realai_client, agent_ids=agent_ids, memory=memory)

        # Run agents in workflow step order
        ordered = [s["agent_id"] for s in steps if isinstance(s, dict) and "agent_id" in s]

        # Seed the first task from the workflow
        first_task = tasks.get(ordered[0], "") if ordered else ""

        # Use run_pipeline with an ordered agent list; the first task seeds it
        # and subsequent agents receive the previous agent's output as their task.
        # To honour per-step tasks, we run each step individually.
        mem = memory or SharedMemory()
        step_results: List[Dict[str, Any]] = []
        current_task = first_task

        for aid in ordered:
            agent = orch.get_agent(aid)
            if agent is None:
                step_results.append({
                    "agent": aid,
                    "task": current_task,
                    "output": "",
                    "success": False,
                    "error": f"Agent '{aid}' not found in team.",
                })
                continue

            # Use the workflow-defined task for the first use of an agent;
            # subsequent passes use the accumulated pipeline output.
            step_task = tasks.get(aid, current_task)
            ctx = mem.get_context()
            result = agent.run(step_task, ctx)
            result["workflow_task"] = step_task
            step_results.append(result)
            mem.store(aid, result)

            if result["success"] and result["output"]:
                current_task = result["output"]

        final_output = ""
        for r in reversed(step_results):
            if r.get("success") and r.get("output"):
                final_output = r["output"]
                break

        return {
            "workflow": workflow.get("name", path.stem),
            "final_output": final_output,
            "steps": step_results,
            "memory": mem.get_context(),
            "success": all(s.get("success") for s in step_results),
        }

    @classmethod
    def list_team(cls, registry: Optional[AgentRegistry] = None) -> List[AgentDefinition]:
        """Return the core RealAI team agent definitions.

        Returns definitions for the agents in :attr:`DEFAULT_PIPELINE` that
        exist in the registry.  Useful for inspecting the team without
        creating an :class:`~orchestrator.Orchestrator`.

        Args:
            registry: Optional pre-loaded registry.

        Returns:
            List of :class:`~registry.AgentDefinition` instances.
        """
        reg = registry or AgentRegistry()
        result: List[AgentDefinition] = []
        for aid in cls.DEFAULT_PIPELINE:
            defn = reg.get(aid)
            if defn is not None:
                result.append(defn)
        return result
