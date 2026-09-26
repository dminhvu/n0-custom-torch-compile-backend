"""M4 · CSE."""

import operator
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.fx as fx
import torch.nn.functional as F

from grader import modules as M
from grader.helpers import assert_close, build, calls, capture, fmt, run
from n0 import passes


def _cse(gm: fx.GraphModule) -> bool:
    changed = passes.cse(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return changed


def _sum_around_write(x):
    y = x.clone()
    before = y.sum()
    y.add_(1)
    after = y.sum()  # same node key as `before`, different value
    return before, after


def _inplace_twice(x):
    y = x.clone()
    y.add_(1)
    y.add_(1)
    return y


x2 = lambda: [torch.randn(2, 2)]  # noqa: E731
xy = lambda: [torch.randn(2, 2), torch.randn(2, 2)]  # noqa: E731

# name -> (fn, inputs, target, how many must stay live, hint)
MERGE: dict[str, tuple[Callable, Callable, Any, int, str]] = {
    "x*2 twice": (lambda x: (x * 2, x * 2), x2, operator.mul, 1, "identical pure nodes"),
    "(x*3).sin() twice": (lambda x: ((x * 3).sin(), (x * 3).sin()), x2, "sin", 1,
                          "the sins only become identical once the muls are merged — does one pass get both?"),
    "cat([x, y]) twice": (lambda x, y: (torch.cat([x, y]), torch.cat([x, y])), xy, torch.cat, 1,
                          "the args contain a list — does your key compare it by value?"),
    "permute([1, 0]) twice": (lambda x: (x.permute([1, 0]), x.permute([1, 0])), x2, "permute", 1,
                              "the args contain a list of ints"),
    "pad(x, [1, 1]) twice": (lambda x: (F.pad(x, [1, 1]), F.pad(x, [1, 1])), x2, "pad", 1, "list inside args"),
}

KEEP: dict[str, tuple[Callable, Callable, Any, int, str]] = {
    "sum(dim=0) vs sum(dim=1)": (lambda x: (x.sum(dim=0), x.sum(dim=1), x.sum(dim=0)), x2, "sum", 2,
                                 "dim=0 and dim=1 differ only in kwargs"),
    "to(float64) vs to(float16)": (lambda x: (x.to(torch.float64), x.to(torch.float16), x.to(torch.float64)), x2, "to", 2,
                                   "the calls differ only in dtype"),
    "i + 1 vs i + 1.0 (int tensor)": (lambda i: (i + 1, i + 1.0), lambda: [torch.arange(4)], operator.add, 2,
                                      "1 == 1.0 and hash(1) == hash(1.0) in Python, but int64 + 1.0 is float32"),
    "x - y vs y - x": (lambda x, y: (x - y, y - x), xy, operator.sub, 2, "argument order matters"),
    "x * 2 vs x * 3": (lambda x: (x * 2, x * 3), x2, operator.mul, 2, "non-Node args are part of the value"),
    "rand(2) twice": (lambda x: (x + torch.rand(2), x + torch.rand(2)), x2, torch.rand, 2,
                      "two draws are two different values"),
    "dropout(training=True) twice": (lambda x: (F.dropout(x, 0.5, True), F.dropout(x, 0.5, True)), x2, F.dropout, 2,
                                     "random masks differ per call"),
}


@pytest.mark.rubric("B3")
@pytest.mark.parametrize("name", list(MERGE))
def test_merges(name):
    """duplicates of a pure computation are merged"""
    fn, inputs, target, want, hint = MERGE[name]
    xs = inputs()
    eager = [t for t in torch.utils._pytree.tree_leaves(fn(*xs))]
    gm, ex = capture(fn, *xs)
    _cse(gm)
    live = calls(gm, target, live_only=True)
    assert len(live) == want, f"{name}: {len(live)} still in use ({fmt(live)}), want {want} — {hint}"
    assert_close(run(gm, ex), eager, name)


@pytest.mark.rubric("B3")
@pytest.mark.parametrize("name", list(KEEP))
def test_keeps_distinct(name):
    """nodes that only look alike are not merged"""
    fn, inputs, target, want, hint = KEEP[name]
    xs = inputs()
    random = target in (torch.rand, F.dropout)
    eager = [t for t in torch.utils._pytree.tree_leaves(fn(*xs))]
    gm, ex = capture(fn, *xs)
    _cse(gm)
    live = calls(gm, target, live_only=True)
    assert len(live) == want, f"{name}: merged down to {len(live)} ({fmt(live)}), must keep {want} — {hint}"
    if not random:
        assert_close(run(gm, ex), eager, name)


@pytest.mark.rubric("B3")
def test_inplace_not_merged():
    """y.add_(1); y.add_(1) look identical but are two separate writes"""
    x = torch.randn(3)
    gm, ex = capture(_inplace_twice, x)
    _cse(gm)
    n = len(calls(gm, "add_"))
    assert n == 2, f"{n} add_ node(s) left — merging two in-place writes drops one of them"
    assert_close(run(gm, ex), [x + 2], "in-place twice")


@pytest.mark.rubric("B3")
def test_not_merged_across_a_write():
    """y.sum(); y.add_(1); y.sum() — the two sums read different values"""
    x = torch.randn(3)
    gm, ex = capture(_sum_around_write, x)
    _cse(gm)
    n = len(calls(gm, "sum", live_only=True))
    assert n == 2, (
        "the two y.sum() calls were merged, but y.add_(1) wrote to y in between — "
        "equal args (same Node) don't mean equal memory contents"
    )
    assert_close(run(gm, ex), [x.sum(), (x + 1).sum()], "sum around an in-place write")


@pytest.mark.rubric("B3")
def test_returns_changed_flag():
    """cse() returns True when it merged something, False otherwise"""
    gm, _, _ = build(next(c for c in M.CASES if c.name == "DupChain"))
    assert _cse(gm) is True, "cse() merged nodes but did not return True"
    assert _cse(gm) is False, "second cse() on the same graph must return False"
    gm, _ = capture(lambda x: x * 2 + 1, torch.randn(3))
    assert _cse(gm) is False, "nothing to merge, but cse() returned True"
