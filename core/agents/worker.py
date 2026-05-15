"""Worker agent."""

import json

from core.agents.base import Agent, AgentContext
from core.agents.safety import AgentSafety
from core.inference.registry import InferenceRegistry
from core.logging.logger import log
from core.metrics.metrics import AGENT_STEPS
from core.tools.executor import ToolExecutionRequest
from core.tools.registry import ToolRegistry
from core.tracing.tracer import tracer


class WorkerAgent(Agent):
    name = "worker"

    def __init__(self, inference: InferenceRegistry, tools: ToolRegistry):
        self.inference = inference
        self.tools = tools
        self.safety = AgentSafety()

    def step(self, messages, context):
        with tracer.start_as_current_span("agent.step"):
            AGENT_STEPS.inc()
            context = context if isinstance(context, dict) else {}
            allowed_tools = [tool.name for tool in self.tools.list()]
            allowed_permissions = context.get("allowed_permissions")
            if not isinstance(allowed_permissions, list):
                allowed_permissions = []
                for tool in self.tools.list():
                    allowed_permissions.extend(getattr(tool, "permissions", []))
                context["allowed_permissions"] = sorted(set(allowed_permissions))

            tool_call = context.get("tool_call")
            if tool_call:
                self.safety.validate_tool_call(tool_call, allowed_tools)
                result = self.tools.execute_tool(
                    tool_call["name"],
                    tool_call.get("arguments", {}),
                    context=context,
                )
                output = {"result": result, "used_tool": tool_call["name"]}
                log("agent.step", {"tool": tool_call["name"]})
                return output

            tool_calls = context.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                plan = _extract_tool_requests(tool_calls, allowed_tools)
                result = self.tools.execute_plan(plan, context=context)
                output = {
                    "results": result,
                    "used_tools": [request.tool_name for request in plan],
                }
                log("agent.step", {"tool_batch": output["used_tools"]})
                return output

            backend = self.inference.get_chat(context["model"])
            response = backend.generate([message.dict() if hasattr(message, "dict") else message for message in messages])
            tool_calls = _extract_tool_calls(response)
            if tool_calls:
                plan = _extract_tool_requests(tool_calls, allowed_tools)
                result = self.tools.execute_plan(plan, context=context)
                output = {
                    "results": result,
                    "used_tools": [request.tool_name for request in plan],
                }
                log("agent.step", {"tool_batch": output["used_tools"]})
                return output
            tool_call = _extract_tool_call(response)
            if tool_call:
                self.safety.validate_tool_call(tool_call, allowed_tools)
                result = self.tools.execute_tool(
                    tool_call["name"],
                    tool_call.get("arguments", {}),
                    context=context,
                )
                output = {"result": result, "used_tool": tool_call["name"]}
                log("agent.step", {"tool": tool_call["name"]})
                return output
            log("agent.step", {"tool": None})
            return {"response": response}


def _extract_tool_call(response):
    choices = response.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    raw = message.get("tool_call")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _extract_tool_calls(response):
    choices = response.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message", {})
    raw = message.get("tool_calls")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None
    return None


def _extract_tool_requests(tool_calls, allowed_tools):
    requests = []
    for index, tool_call in enumerate(tool_calls, start=1):
        if not isinstance(tool_call, dict):
            raise ValueError("Invalid tool call payload.")
        name = tool_call.get("name")
        if name not in set(allowed_tools or []):
            raise PermissionError("Tool not allowed")
        args = tool_call.get("arguments", {})
        if not isinstance(args, dict):
            args = {}
        depends_on = tool_call.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []
        requests.append(ToolExecutionRequest(
            request_id=str(tool_call.get("id", "tool-call-{0}".format(index))),
            tool_name=name,
            arguments=args,
            depends_on=[str(dep) for dep in depends_on],
        ))
    return requests
