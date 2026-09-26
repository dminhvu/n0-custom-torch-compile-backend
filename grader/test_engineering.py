"""C · engineering. C3 (README, commit history) is reviewed by hand."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CORE = sorted((ROOT / "n0").glob("*.py"))
TESTS = ROOT / "tests"
MAX_CORE_LINES = 300
OWN_TESTS_LIMIT_S = 120


def _functions() -> list[tuple[Path, ast.FunctionDef]]:
    out = []
    for path in CORE:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append((path, node))
    return out


def _require_implemented() -> None:
    stubs = [
        f.name for _, f in _functions()
        if len(f.body) == 1 and isinstance(f.body[0], ast.Raise) and "NotImplementedError" in ast.unparse(f.body[0])
    ]
    if stubs:
        pytest.fail(f"not implemented yet: {', '.join(stubs)} — engineering is graded on finished code")


@pytest.mark.rubric("C1")
def test_core_size():
    """core (n0/*.py) ≤ ~300 lines of code"""
    _require_implemented()
    n = sum(
        1 for p in CORE for line in p.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert n <= MAX_CORE_LINES, f"n0/ has {n} non-blank, non-comment lines (limit ~{MAX_CORE_LINES})"


@pytest.mark.rubric("C1")
def test_one_function_per_pass():
    """each pass is one plain function over the graph"""
    _require_implemented()
    import inspect

    from n0 import passes

    for name in ("dce", "cse", "fold_constants", "optimize"):
        assert inspect.isfunction(getattr(passes, name, None)), f"passes.{name} is not a plain function"


@pytest.mark.rubric("C1")
def test_type_hints():
    """every function in n0/ has parameter and return annotations"""
    _require_implemented()
    missing = []
    for path, f in _functions():
        params = [a for a in f.args.posonlyargs + f.args.args + f.args.kwonlyargs if a.arg not in ("self", "cls")]
        if any(a.annotation is None for a in params) or f.returns is None:
            missing.append(f"{path.name}:{f.lineno} {f.name}")
    assert not missing, f"missing type hints: {', '.join(missing)}"


@pytest.mark.rubric("C2")
def test_one_test_file_per_pass():
    """tests/ has a test file for dce, fold and cse"""
    files = [p.name for p in TESTS.glob("test_*.py")] if TESTS.is_dir() else []
    missing = [k for k in ("dce", "fold", "cse") if not any(k in f for f in files)]
    assert not missing, f"no tests/test_*{{{','.join(missing)}}}*.py yet (have: {files or 'nothing'})"


@pytest.mark.rubric("C2")
def test_integration_test():
    """tests/ has an integration test that goes through torch.compile"""
    hits = [p.name for p in TESTS.glob("test_*.py") if "torch.compile" in p.read_text()] if TESTS.is_dir() else []
    assert hits, "no test in tests/ calls torch.compile(..., backend=...)"


@pytest.mark.rubric("C2")
@pytest.mark.time_limit(OWN_TESTS_LIMIT_S + 10)
def test_pytest_green():
    """uv run pytest (your tests/) is green"""
    assert TESTS.is_dir(), "no tests/ directory yet"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
            cwd=ROOT, capture_output=True, text=True, timeout=OWN_TESTS_LIMIT_S,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"your test suite ran over {OWN_TESTS_LIMIT_S}s — probably the same non-terminating loop")
    tail = "\n".join(proc.stdout.strip().splitlines()[-5:])
    assert proc.returncode != 5, "tests/ contains no tests"
    assert proc.returncode == 0, f"your test suite is red:\n{tail}"
