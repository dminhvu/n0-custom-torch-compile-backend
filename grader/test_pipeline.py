"""M5 · fixpoint driver — every test module end to end."""

import pytest

from grader import modules as M
from grader.helpers import assert_close, build, calls, fmt, run
from n0 import passes


@pytest.mark.rubric("A1")
@pytest.mark.parametrize("case", M.CASES, ids=lambda c: c.name)
@pytest.mark.fixpoint
def test_matches_eager(case):
    """optimize(): output allclose to eager (on two consecutive calls)"""
    gm, ex, eager = build(case)
    assert passes.optimize(gm) is gm, "optimize() must return the GraphModule it optimized"
    gm.graph.lint()
    assert gm.code.strip() == gm.graph.python_code("self").src.strip(), (
        "gm.code is stale — the graph changed but gm.recompile() was not called, so forward() runs the old code"
    )
    assert_close(run(gm, ex), eager, case.name)
    assert_close(run(gm, ex), eager, f"{case.name} (2nd call)")


@pytest.mark.rubric("A1")
@pytest.mark.parametrize("case", M.CASES, ids=lambda c: c.name)
@pytest.mark.fixpoint
def test_removes_expected_work(case):
    """optimize(): the known dead/constant/duplicate work is gone"""
    gm, _, _ = build(case)
    passes.optimize(gm)
    left = calls(gm)
    assert len(left) <= case.max_calls, (
        f"{case.name}: {len(left)} call nodes left, expected ≤ {case.max_calls}: {fmt(left)}"
    )


@pytest.mark.rubric("A2")
@pytest.mark.parametrize("changes_in", ["fold_constants", "cse", "dce"])
def test_fixpoint_loop(monkeypatch, changes_in):
    """optimize() keeps going until all three passes report no change"""
    log: list[tuple[str, bool]] = []

    def stub(name: str):
        def run_pass(*_args: object, **_kwargs: object) -> bool:
            changed = name == changes_in and not any(n == name for n, _ in log)  # 1st call only
            log.append((name, changed))
            return changed
        return run_pass

    for name in ("fold_constants", "cse", "dce"):
        monkeypatch.setattr(passes, name, stub(name))
    gm, _, _ = build(M.CASES[0])
    passes.optimize(gm)
    assert log, (
        "the grader replaces passes.fold_constants/cse/dce with stubs to watch the loop, but optimize() "
        "never called them — call the passes by their module-level names (not via a list built at import time)"
    )
    calls_ = [n for n, _ in log]
    last_change = max(i for i, (_, changed) in enumerate(log) if changed)
    quiet_after = {n for n, _ in log[last_change + 1 :]}
    assert quiet_after == {"fold_constants", "cse", "dce"}, (
        f"only {changes_in}() reported a change (its first call); calls were {calls_}. After the last "
        "change, every pass must run once more and report nothing — otherwise it isn't a fixpoint"
    )
    assert len(log) <= 12, f"{len(log)} pass calls for one change — the loop doesn't stop once nothing changes"


@pytest.mark.rubric("A2")
@pytest.mark.parametrize("case", M.CASES, ids=lambda c: c.name)
@pytest.mark.fixpoint
def test_idempotent(case):
    """after optimize(), no pass changes anything and a second optimize() is a no-op"""
    gm, _, _ = build(case)
    passes.optimize(gm)
    code = gm.code
    for name, step in (
        ("fold_constants", lambda: passes.fold_constants(gm)),
        ("cse", lambda: passes.cse(gm.graph)),
        ("dce", lambda: passes.dce(gm.graph)),
    ):
        assert step() is False, f"{name}() still found work after optimize() — not a fixpoint"
        gm.recompile()
        assert gm.code == code, f"{name}() changed an already-optimized graph but returned False"
    passes.optimize(gm)
    assert gm.code == code, "a second optimize() changed the graph"
