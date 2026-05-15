"""Tool registry."""

from typing import Any, Dict, List

from core.tools.base import Tool
from core.tools.credentials import ToolCredentialManager
from core.tools.executor import ToolExecutionEngine, ToolExecutionRequest
from core.metrics.metrics import TOOL_CALLS


class ToolRegistry:
    def __init__(
        self,
        executor: ToolExecutionEngine = None,
        credential_manager: ToolCredentialManager = None,
    ):
        self.tools: Dict[str, Tool] = {}
        self.credential_manager = credential_manager or ToolCredentialManager()
        self.executor = executor or ToolExecutionEngine(credential_manager=self.credential_manager)

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self.tools:
            raise KeyError("Unknown tool: {0}".format(name))
        return self.tools[name]

    def list(self):
        return list(self.tools.values())

    def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any] = None):
        tool = self.get(name)
        result = self.executor.execute_tool(
            name,
            tool,
            args or {},
            context=context if isinstance(context, dict) else {},
        )
        TOOL_CALLS.labels(tool=name).inc()
        if result.status == "success":
            return result.output
        if result.status == "timeout":
            raise TimeoutError(result.error or "Tool execution timed out")
        raise RuntimeError(result.error or "Tool execution failed")

    def execute_plan(
        self,
        requests: List[ToolExecutionRequest],
        context: Dict[str, Any] = None,
    ) -> Dict[str, Dict[str, Any]]:
        results = self.executor.execute_plan(
            requests,
            self.tools,
            context=context if isinstance(context, dict) else {},
        )
        for request in requests:
            TOOL_CALLS.labels(tool=request.tool_name).inc()
        return {
            request_id: result.to_dict()
            for request_id, result in results.items()
        }

    def get_audit_log(self):
        return self.executor.get_audit_log()
