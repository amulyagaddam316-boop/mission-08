"""Traceback diagnosis and bounded, test-grounded repair loop."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Any


@dataclass(frozen=True)
class Diagnostic:
    file: str | None
    line_number: int | None
    exception_type: str
    fault_summary: str


# Match Python traceback frames such as:
# File "/some/path/example.py", line 12, in <module>
_TRACE_FILE_RE = re.compile(r'File "([^"]+)", line (\d+)(?:, in .*)?')

# Match both exceptions with messages (ValueError: bad input) and
# exceptions without messages (AssertionError).
_EXCEPTION_RE = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_.]*)(?::\s*(.*))?$"
)


def parse_traceback(stderr_text: str) -> Diagnostic:
    """Extract the final traceback frame and exception from stderr text."""
    frames = list(_TRACE_FILE_RE.finditer(stderr_text))
    file_name = frames[-1].group(1) if frames else None
    line_number = int(frames[-1].group(2)) if frames else None

    exception_type = "UnknownError"
    fault_summary = "No error details were provided."

    # The exception is normally the last non-empty line in a Python traceback.
    lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        match = _EXCEPTION_RE.match(last_line)
        if match:
            exception_type = match.group(1).split(".")[-1]
            fault_summary = (match.group(2) or "").strip()
        else:
            fault_summary = last_line

    return Diagnostic(
        file=file_name,
        line_number=line_number,
        exception_type=exception_type,
        fault_summary=fault_summary,
    )


def generate_verbal_reflection(
    diag: Diagnostic,
    failed_code: str,
) -> str:
    """Create a reflection grounded in the observed diagnostic and source."""
    location = (
        f"{diag.file or 'unknown file'}"
        + (f", line {diag.line_number}" if diag.line_number is not None else "")
    )

    nearby = ""
    if diag.line_number is not None:
        lines = failed_code.splitlines()
        start = max(0, diag.line_number - 2)
        end = min(len(lines), diag.line_number + 1)
        nearby_lines = [
            f"{index + 1}: {lines[index]}"
            for index in range(start, end)
        ]
        if nearby_lines:
            nearby = "\nNearby source:\n" + "\n".join(nearby_lines)

    return (
        f"The test run reported {diag.exception_type} at {location}: "
        f"{diag.fault_summary}. Inspect the operation at that location and "
        "verify its input types, boundary conditions, and required variables "
        "before changing the code. Make the smallest targeted patch, then "
        f"rerun the same tests.{nearby}"
    )


def _normalise_code(code: str) -> str:
    """Normalize whitespace before hashing candidate code."""
    return "".join(code.split())


def is_duplicate_patch(
    candidate_code: str,
    seen_hashes: set[str] | None = None,
) -> bool:
    """Detect repeated code using SHA-256.

    Supply the same set on each call to track patches across a repair session.
    """
    normalized = _normalise_code(candidate_code)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    if seen_hashes is None:
        # Without a shared set there is no history to compare against.
        return False

    if digest in seen_hashes:
        return True

    seen_hashes.add(digest)
    return False


def run_repair_loop(
    initial_code: str,
    tests: Callable[[str], bool],
    patcher: Callable[[str, str, int], str | None],
    *,
    max_trials: int = 5,
) -> dict[str, Any]:
    """Run tests, reflect on failures, and request patches until tests pass.

    tests(code):
        Return True if all tests pass, False otherwise. It may also raise an
        exception; the exception text is used as diagnostic input.

    patcher(code, reflection, trial):
        Return revised source code, or None if no patch is available.

    The episodic reflection buffer retains at most the latest three reflections.
    """
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1.")

    code = initial_code
    reflections: deque[str] = deque(maxlen=3)
    seen_hashes: set[str] = set()
    history: list[dict[str, Any]] = []

    for trial in range(1, max_trials + 1):
        try:
            passed = bool(tests(code))
            error_text = ""
        except Exception as exc:
            passed = False
            error_text = f"{type(exc).__name__}: {exc}"

        if passed:
            return {
                "success": True,
                "code": code,
                "trials": trial - 1,
                "reflections": list(reflections),
                "history": history,
            }

        if error_text:
            diagnostic = parse_traceback(error_text)
        else:
            diagnostic = Diagnostic(
                file=None,
                line_number=None,
                exception_type="TestFailure",
                fault_summary="The supplied tests returned False.",
            )

        reflection = generate_verbal_reflection(diagnostic, code)
        reflections.append(reflection)
        history.append({
            "trial": trial,
            "diagnostic": diagnostic,
            "reflection": reflection,
        })

        if trial == max_trials:
            break

        candidate = patcher(code, reflection, trial)
        if candidate is None:
            break

        if is_duplicate_patch(candidate, seen_hashes):
            history.append({
                "trial": trial,
                "status": "duplicate patch rejected",
            })
            break

        code = candidate

    return {
        "success": False,
        "code": code,
        "trials": len(history),
        "reflections": list(reflections),
        "history": history,
    }


def run_python_tests(
    code: str,
    test_code: str,
    timeout: int = 10,
) -> tuple[bool, str]:
    """Run candidate plus test code in a temporary Python file.

    WARNING: This helper executes code. Use only with code you trust. It is
    not a secure sandbox and must not be used to execute untrusted code.
    """
    combined = code + "\n\n" + test_code

    with tempfile.TemporaryDirectory() as temp_dir:
        test_path = Path(temp_dir) / "candidate_test.py"
        test_path.write_text(combined, encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    return result.returncode == 0, result.stderr


if __name__ == "__main__":
    # This example intentionally contains a bug: subtraction instead of addition.
    sample_code = (
        "def add(a, b):\n"
        "    return a - b\n"
    )
    sample_tests = "assert add(2, 3) == 5\n"

    tests_passed, error_output = run_python_tests(sample_code, sample_tests)

    print("Tests passed:", tests_passed)
    if error_output:
        print("Diagnostic:", parse_traceback(error_output))
    else:
        print("Diagnostic: No error")
