"""AST-based static checks and episodic-memory provenance validation.

This is a static policy checker, not a complete operating-system sandbox.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any


FORBIDDEN_MODULES = {"os", "subprocess", "socket", "shutil", "sys", "ctypes"}
DANGEROUS_BUILTINS = {"eval", "exec", "compile", "__import__", "open"}
FORBIDDEN_ATTRIBUTES = {"__subclasses__", "__globals__", "__builtins__", "__class__"}


@dataclass
class SandboxGuardian(ast.NodeVisitor):
    violations: list[str] = field(default_factory=list)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            if root in FORBIDDEN_MODULES:
                self.violations.append(
                    f"Forbidden import '{alias.name}' at line {node.lineno}"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = (node.module or "").split(".", 1)[0]
        if module in FORBIDDEN_MODULES:
            self.violations.append(
                f"Forbidden import from '{node.module}' at line {node.lineno}"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in DANGEROUS_BUILTINS:
            self.violations.append(
                f"Dangerous call '{node.func.id}()' at line {node.lineno}"
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.violations.append(
                f"Forbidden attribute '{node.attr}' at line {node.lineno}"
            )
        self.generic_visit(node)


def inspect_code(source_code: str) -> dict[str, Any]:
    """Return whether source is allowed by the static policy and list violations."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return {
            "safe": False,
            "violations": [f"SyntaxError at line {exc.lineno}: {exc.msg}"],
        }
    guardian = SandboxGuardian()
    guardian.visit(tree)
    return {"safe": not guardian.violations, "violations": guardian.violations}


_BYPASS_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disable\s+(the\s+)?(safety|security|sandbox)",
    r"bypass\s+(the\s+)?(security|safety|permission)",
    r"grant\s+.*(admin|root|privilege)",
    r"exfiltrat(e|ion)",
]


def audit_episodic_write(
    content: str,
    trust_score: float,
    provenance: str = "VERIFIED",
) -> dict[str, Any]:
    """Reject low-trust/untrusted provenance and embedded privilege-bypass directives."""
    reasons: list[str] = []
    if trust_score < 0.75:
        reasons.append(f"trust_score {trust_score:.2f} is below required 0.75")
    if provenance.strip().upper() == "UNTRUSTED_TELEMETRY":
        reasons.append("provenance is UNTRUSTED_TELEMETRY")
    for pattern in _BYPASS_PATTERNS:
        if re.search(pattern, content, flags=re.IGNORECASE):
            reasons.append("content contains a privilege-escalation or safety-bypass directive")
            break
    return {"accepted": not reasons, "reasons": reasons}


if __name__ == "__main__":
    examples = [
        "x = 1 + 2",
        "import os\nos.system('echo unsafe')",
        "value = eval('2 + 2')",
        "print(obj.__globals__)",
    ]
    for example in examples:
        print(example, "=>", inspect_code(example))
    print(audit_episodic_write("Remember the project deadline.", 0.9, "VERIFIED"))
