"""Code execution tool with local sandbox limits."""

from typing import Any, Dict

from core.security.python_sandbox import PythonSandbox
from core.tools.base import Tool
from core.tools.permissions import Permissions


class CodeExecutionTool(Tool):
    name = "code_exec"
    description = "Run Python code in a constrained local sandbox"
    params_schema = {"code": {"type": "string"}}
    permissions = [Permissions.CODE_EXEC]
    requires_sandbox = True
    sandbox_type = "python"

    def __init__(self, sandbox: PythonSandbox = None):
        self.sandbox = sandbox or PythonSandbox()

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        code = str(kwargs.get("code", ""))
        return self.sandbox.run(code)

    def build_sandbox_script(self, arguments: Dict[str, Any] = None, context: Dict[str, Any] = None) -> str:
        """Build sandbox input code for execution."""
        arguments = arguments if isinstance(arguments, dict) else {}
        return str(arguments.get("code", ""))

    def parse_sandbox_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize sandbox output."""
        return dict(result if isinstance(result, dict) else {"result": result})
