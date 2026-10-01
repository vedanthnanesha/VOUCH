"""Code interventions and a subprocess test-runner sandbox for the MBPP
scope-check.

The interventions modify a candidate Python solution:
  - remove_line(k):  drop line k (1-indexed) from the candidate
  - swap_lines(k, j): swap two lines
  - ablate_function(name): replace function body with `return None`

The test runner takes the candidate code (possibly modified), prepends
the test_setup_code, appends each test assertion, and runs each one in
an isolated subprocess with a strict time limit. Returns the number of
tests that passed and the per-test pass/fail vector.

The sandbox uses subprocess.run with a hard timeout; it is NOT a
security boundary. Do not run untrusted code under this.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


_RUNNER_SCRIPT = r"""
import json, sys, traceback, signal

def _alarm(signum, frame):
    raise TimeoutError("test timed out")

signal.signal(signal.SIGALRM, _alarm)
signal.alarm(5)

payload = json.loads(sys.stdin.read())
code = payload["code"]
setup = payload.get("setup", "")
tests = payload.get("tests", [])

ns = {}
results = []
ok_compile = True
try:
    exec(setup + "\n" + code, ns, ns)
except Exception as e:
    ok_compile = False
    err_global = f"compile/exec error: {type(e).__name__}: {e}"
else:
    err_global = None

for t in tests:
    if not ok_compile:
        results.append({"ok": False, "error": err_global})
        continue
    try:
        signal.alarm(5)
        exec(t, ns, ns)
        results.append({"ok": True, "error": None})
    except Exception as e:
        results.append({"ok": False,
                         "error": f"{type(e).__name__}: {e}"})

print(json.dumps({"compile_ok": ok_compile, "results": results}), flush=True)
"""


# -------- intervention functions --------

def remove_line(code: str, line_index_1based: int) -> str:
    lines = code.splitlines()
    if not lines:
        return code
    i = max(1, min(line_index_1based, len(lines)))
    return "\n".join(lines[:i - 1] + lines[i:])


def swap_lines(code: str, i: int, j: int) -> str:
    lines = code.splitlines()
    if not lines:
        return code
    a = max(1, min(i, len(lines))) - 1
    b = max(1, min(j, len(lines))) - 1
    lines[a], lines[b] = lines[b], lines[a]
    return "\n".join(lines)


def ablate_function(code: str, func_name: str) -> str:
    """Replace the body of ``func_name`` with ``return None``. If
    ``func_name`` does not appear, return code unchanged.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    out_lines = code.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            body_start = node.body[0].lineno - 1
            body_end = node.body[-1].end_lineno
            indent = " " * (node.col_offset + 4)
            out_lines = (
                out_lines[:body_start]
                + [f"{indent}return None"]
                + out_lines[body_end:]
            )
            break
    return "\n".join(out_lines)


# -------- sandbox --------

def run_tests(code: str, tests: list[str], setup: str = "",
              python: Optional[str] = None,
              timeout: float = 15.0) -> dict:
    """Return {compile_ok: bool, results: list[{ok, error}]}.

    Each test runs in its own subprocess call with a 5s SIGALRM inside
    the runner and a 15s outer subprocess timeout.
    """
    python = python or sys.executable
    payload = {"code": code, "setup": setup, "tests": tests}
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_RUNNER_SCRIPT)
        runner_path = f.name
    try:
        proc = subprocess.run(
            [python, runner_path],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return {"compile_ok": False, "results": [
                {"ok": False, "error": f"runner rc={proc.returncode}"}
                for _ in tests]}
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as e:
            return {"compile_ok": False, "results": [
                {"ok": False, "error": f"bad runner output: {e}"}
                for _ in tests]}
    except subprocess.TimeoutExpired:
        return {"compile_ok": False, "results": [
            {"ok": False, "error": "outer timeout"} for _ in tests]}
    finally:
        try:
            os.unlink(runner_path)
        except OSError:
            pass


def pass_signature(test_results: list[dict]) -> str:
    """Compact signature: e.g. 'pass=2,fail=1' or per-test 'TFT'."""
    return "".join("T" if r.get("ok") else "F" for r in test_results)
