"""M2 · DCE."""

import operator

import pytest
import torch
import torch.fx as fx

from grader import modules as M
from grader.helpers import assert_close, build, calls, capture, fmt, run, snapshot, trace
from n0 import passes


def _dce(gm: fx.GraphModule) -> bool:
    changed = passes.dce(gm.graph)
    gm.graph.lint()
    gm.recompile()
    return changed


def _case(name: str) -> M.Case:
    return next(c for c in M.CASES if c.name == name)


@pytest.mark.rubric("B1")
def test_dead_chain_one_call():
    """a dead chain a→b→c disappears in a single dce() call"""
    gm, ex, eager = build(_case("DeadChain"))
    _dce(gm)
    left = calls(gm)
    assert len(left) == 1, (
        f"after one dce() call these remain: {fmt(left)} — only relu is live. "
        "Erasing a node can make its producers dead: which walk order catches that in one pass?"
    )
    assert_close(run(gm, ex), eager, "DeadChain")


@pytest.mark.rubric("B1")
def test_dead_feeds_dead_one_call():
    """a dead node whose only user is another dead node goes in the same call"""
    gm, ex, eager = build(_case("DeadFeedsDead"))
    _dce(gm)
    left = calls(gm)
    assert len(left) == 2, (
        f"remaining: {fmt(left)} — only sin and mul are live. dead1 has a user (dead2) when a "
        "forward walk reaches it; think about the order in which users become empty"
    )


@pytest.mark.rubric("B1")
def test_returns_changed_flag():
    """dce() returns True when it erased something, False when there was nothing to erase"""
    gm, _, _ = build(_case("DeadChain"))
    assert _dce(gm) is True, "dce() erased nodes but did not return True"
    assert _dce(gm) is False, "second dce() on a clean graph must return False (the fixpoint loop relies on it)"


@pytest.mark.rubric("B1")
def test_placeholders_and_output_kept():
    """unused placeholders and the output node are never erased"""
    gm = trace(M.UnusedInput())
    _dce(gm)
    ops = [n.op for n in gm.graph.nodes]
    assert ops.count("placeholder") == 2, "an unused input was erased — that changes the function's signature"
    assert ops.count("output") == 1, "the output node was erased"
    x = torch.randn(3)
    assert torch.allclose(gm(x, torch.randn(3)), x + 1)


@pytest.mark.rubric("B1")
def test_side_effectful_function_kept():
    """torch._assert has no users but is side-effectful — it stays"""
    g = fx.Graph()
    x = g.placeholder("x")
    total = g.call_method("sum", (x,))
    ok = g.call_function(operator.gt, (total, -1e9))
    g.call_function(torch._assert, (ok, "must hold"))
    g.output(x)
    gm = fx.GraphModule(torch.nn.Module(), g)
    _dce(gm)
    assert calls(gm, torch._assert), (
        "torch._assert was erased. Its effect is raising, not a value — see "
        "torch.fx.node._side_effectful_functions / Node.is_impure()"
    )


@pytest.mark.rubric("B1")
def test_setitem_kept():
    """y[0] = 5 (operator.setitem: in-place, no '_' suffix, zero users) stays"""
    gm, ex, eager = build(_case("SetItem"))
    _dce(gm)
    assert calls(gm, operator.setitem), "operator.setitem was erased — it writes into y, which the output reads"
    assert_close(run(gm, ex), eager, "SetItem")


@pytest.mark.rubric("B1")
def test_no_special_casing_by_name():
    """decisions depend on op/target, never on node.name"""
    g = fx.Graph()
    x = g.placeholder("x")
    y = g.call_method("clone", (x,))
    g.create_node("call_method", "mul_", (y, 2), name="scale")  # in-place, harmless name
    g.create_node("call_function", torch.add, (x, 1), name="add_")  # pure + dead, suspicious name
    g.output(y)
    gm = fx.GraphModule(torch.nn.Module(), g)
    _dce(gm)
    names = {n.name for n in gm.graph.nodes}
    assert "scale" in names, "in-place y.mul_(2) (node named 'scale') was erased — the target says in-place, not the name"
    assert not any(n.target is torch.add for n in gm.graph.nodes), (
        "a dead torch.add named 'add_' survived — node names are generated, only op/target carry meaning"
    )


@pytest.mark.rubric("A3")
def test_inplace_method_survives():
    """y.add_(1); y.mul_(2) survive DCE and the result is still right"""
    gm, ex, eager = build(_case("InPlace"))
    _dce(gm)
    kept = {n.target for n in calls(gm)}
    assert {"add_", "mul_"} <= kept, f"in-place op(s) erased; remaining: {fmt(calls(gm))}"
    assert not calls(gm, operator.mul), "dead x * 7 was kept"
    assert_close(run(gm, ex), eager, "InPlace")


@pytest.mark.rubric("A3")
def test_buffer_copy_survives():
    """self.state.copy_(x) survives DCE — the buffer is still updated"""
    module = M.BufferCopy()
    x = torch.randn(3)
    gm, ex = capture(module, x)
    _dce(gm)
    assert calls(gm, "copy_"), "state.copy_(x) was erased — its effect is on the buffer, not on a return value"
    args = [t.clone() for t in ex]
    gm(*args)
    state = next(a for a, n in zip(args, gm.graph.find_nodes(op="placeholder")) if "state" in n.target)
    assert torch.equal(state, x), "the buffer was not updated after DCE"


def _is_inplace(n: fx.Node) -> bool:
    name = n.target if isinstance(n.target, str) else getattr(n.target, "__name__", "")
    return name.endswith("_") and not name.endswith("__")


@pytest.mark.rubric("A4")
@pytest.mark.parametrize("name", ["DeadChain", "DeadFeedsDead", "InPlace", "SetItem", "BufferCopy", "MLP[dynamo]", "MLP[fx]"])
def test_reference_finds_nothing(name):
    """torch's eliminate_dead_code() finds nothing left after your DCE"""
    gm, _, _ = build(_case(name))
    _dce(gm)
    ref = snapshot(gm)
    before = {n.name: n for n in ref.graph.nodes}
    ref.graph.eliminate_dead_code()
    after = {n.name for n in ref.graph.nodes}
    # torch's reference treats in-place call_method/call_function as pure and would erase them;
    # that is a bug in the reference for non-functional graphs, so those don't count against you.
    missed = [n for name_, n in before.items() if name_ not in after and not _is_inplace(n)]
    assert not missed, f"your DCE left dead nodes the reference removes: {fmt(missed)}"
