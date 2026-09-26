"""M3 · constant folding."""

import inspect

import pytest
import torch
import torch.fx as fx

from grader import modules as M
from grader.helpers import assert_close, build, calls, capture, fmt, get_attrs, run, trace
from n0 import passes

MAX_TESTABLE_CAP = 2**24  # 64 MB of float32 — beyond that a cap no longer bounds anything


def _fold(gm: fx.GraphModule) -> bool:
    changed = passes.fold_constants(gm)
    gm.graph.lint()
    gm.recompile()
    return changed


def _case(name: str) -> M.Case:
    return next(c for c in M.CASES if c.name == name)


@pytest.mark.rubric("B2")
def test_const_chain():
    """torch.ones(3, 3) * 2 + 1 folds to a single get_attr, recursively"""
    gm, ex, eager = build(_case("ConstChain"))
    assert _fold(gm) is True, "fold_constants changed the graph but did not return True"
    live = calls(gm, live_only=True)
    assert len(live) == 1, (
        f"still computed at runtime: {fmt(live)} — only x + <const> should be. "
        "ones(...) has no tensor inputs; mul's input is ones; add's is mul: constness propagates"
    )
    assert len(get_attrs(gm)) == 1, f"expected one folded constant, got {fmt(get_attrs(gm))}"
    assert_close(run(gm, ex), eager, "ConstChain")


@pytest.mark.rubric("B2")
def test_folded_values_are_registered_buffers():
    """folded results are registered buffers read through get_attr"""
    gm, _, _ = build(_case("ConstChain"))
    _fold(gm)
    buffers = dict(gm.named_buffers())
    for n in get_attrs(gm):
        assert n.target in buffers, (
            f"get_attr %{n.name} → '{n.target}' is not a registered buffer "
            "(register_buffer makes it follow .to(device) and appear in state_dict)"
        )


@pytest.mark.rubric("B2")
def test_parameter_subexpression():
    """w.t() * 2 + b over parameters/buffers (get_attr) folds — freezing"""
    gm, ex, eager = build(_case("ParamOnly[fx]"))
    _fold(gm)
    live = calls(gm, live_only=True)
    assert len(live) == 1, f"still computed at runtime: {fmt(live)} — only x @ <const> depends on the input"
    assert_close(run(gm, ex), eager, "ParamOnly")


@pytest.mark.rubric("B2")
def test_constants_inside_list_args():
    """torch.cat([ones, zeros]) is constant: look inside list/tuple args"""
    gm, ex, eager = build(_case("ConstList"))
    _fold(gm)
    live = calls(gm, live_only=True)
    assert len(live) == 1, f"still computed at runtime: {fmt(live)} — the cat's inputs sit inside a list"
    assert_close(run(gm, ex), eager, "ConstList")


@pytest.mark.rubric("B2")
def test_list_with_input_not_folded():
    """torch.cat([x, ones]) is not constant (x hides inside the list)"""
    gm, ex, eager = build(_case("MixedList"))
    _fold(gm)
    assert calls(gm, torch.cat, live_only=True), "cat([x, ...]) was folded although x is an input"
    assert not calls(gm, torch.ones, live_only=True), "torch.ones(2) inside the list was not folded"
    assert_close(run(gm, ex), eager, "MixedList")


@pytest.mark.rubric("B2")
def test_input_dependent_not_folded():
    """nothing downstream of a placeholder is folded; returns False when nothing folds"""
    gm, ex = capture(lambda x: x * 2 + 1, torch.randn(3))
    assert _fold(gm) is False, "nothing is constant here, but fold_constants returned True"
    assert len(calls(gm, live_only=True)) == 2


@pytest.mark.rubric("B2")
def test_dynamo_params_are_placeholders():
    """under Dynamo, parameters arrive as placeholders — not constants, nothing folds"""
    gm, ex, eager = build(M.Case("ParamOnly[dynamo]", M.ParamOnly, lambda: [torch.randn(3, 4)], "dynamo", 4))
    assert _fold(gm) is False, "a node depending on a placeholder (a lifted parameter) was folded"
    assert_close(run(gm, ex), eager, "ParamOnly[dynamo]")


@pytest.mark.rubric("B2")
def test_nondeterministic_not_folded():
    """random ops (rand, randn, randint, randperm, rand_like, bernoulli, dropout) are never folded"""
    folded = []
    for op, (fn, target) in M.RANDOM_OPS.items():
        gm, _ = capture(fn, torch.randn(3))
        _fold(gm)
        if not calls(gm, target, live_only=True):
            folded.append(op)
    assert not folded, (
        f"folded: {', '.join(folded)} — every call must draw fresh random numbers; a folded one "
        "returns the same tensor forever"
    )


@pytest.mark.rubric("B2")
def test_size_cap_respected():
    """a constant bigger than FOLD_SIZE_CAP stays a runtime computation"""
    cap = getattr(passes, "FOLD_SIZE_CAP", None)
    assert isinstance(cap, int) and cap > 0, "define FOLD_SIZE_CAP: int > 0 in n0/passes.py (in elements)"
    assert cap <= MAX_TESTABLE_CAP, f"FOLD_SIZE_CAP = {cap} elements is effectively no cap (grader limit {MAX_TESTABLE_CAP})"
    gm, ex = capture(M.BigConst(cap + 1), torch.randn(cap + 1))
    _fold(gm)
    assert calls(gm, torch.ones, live_only=True), f"torch.ones({cap + 1}) was folded despite FOLD_SIZE_CAP = {cap}"
    at_cap = min(cap, 4096)
    gm, ex = capture(M.BigConst(at_cap), torch.randn(at_cap))
    _fold(gm)
    assert not calls(gm, torch.ones, live_only=True), (
        f"torch.ones({at_cap}) was not folded although {at_cap} ≤ FOLD_SIZE_CAP = {cap} — "
        "is the comparison > rather than >=, and is the cap counted in elements, not bytes?"
    )


@pytest.mark.rubric("B2")
def test_size_cap_justified():
    """FOLD_SIZE_CAP carries a comment justifying the value"""
    lines = inspect.getsource(passes).splitlines()
    idx = next((i for i, l in enumerate(lines) if l.lstrip().startswith("FOLD_SIZE_CAP")), None)
    assert idx is not None, "FOLD_SIZE_CAP is not defined at module level"
    context = [l for l in lines[max(0, idx - 4) : idx + 1] if "#" in l]
    text = " ".join(l.split("#", 1)[1] for l in context).strip()
    assert text and "TODO" not in text and len(text.split()) >= 6, (
        "justify the cap in a comment on/above FOLD_SIZE_CAP (why this number? what does it trade?)"
    )


@pytest.mark.rubric("A1")
def test_non_tensor_constants():
    """self.w.shape[0] (an int, not a tensor) doesn't crash folding"""
    gm, ex, eager = build(_case("ShapeOfParam[fx]"))
    _fold(gm)
    assert_close(run(gm, ex), eager, "ShapeOfParam")


@pytest.mark.rubric("A1")
def test_mutated_constant():
    """t = zeros(3); t.add_(1): still right on the 2nd and 3rd call"""
    gm, ex, eager = build(_case("MutatedConst"))
    _fold(gm)
    assert_close(run(gm, ex), eager, "MutatedConst, call #1")
    for i in (2, 3):
        assert_close(
            run(gm, ex), eager,
            f"MutatedConst, call #{i} (call #1 was right — something persists between calls; "
            "is a tensor that gets written to really a constant?)",
        )
