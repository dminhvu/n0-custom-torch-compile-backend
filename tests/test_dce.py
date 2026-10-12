import torch
import torch.fx as fx

from n0.passes import dce

INPUT_SIZE = 100

def count_calls(gm: fx.GraphModule) -> int:
    """Number of call_* nodes: the ones DCE is allowed to remove."""
    return sum(n.op.startswith("call_") for n in list(gm.graph.nodes))

class DeadChain(torch.nn.Module):
    """3 call nodes are dead (sin -> cos -> mul, result unused), 1 is live (add)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dead = x.sin().cos() * 2  # noqa: F841 — the point of the test
        return x + 1


def test_dead_chain_removed():
    model = DeadChain().eval()
    gm = fx.symbolic_trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x) 
    before = count_calls(gm)

    changed = dce(gm.graph)
    gm.recompile()

    assert changed
    assert before - count_calls(gm) == 3
    assert torch.allclose(gm(x), expected)


class InplaceNoUsers(torch.nn.Module):
    """In-place method add_() on `y` should survive, even though nothing catches y."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        y.add_(1)
        return y


def test_inplace_survives():
    model = InplaceNoUsers().eval()
    gm = fx.symbolic_trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x)

    changed = dce(gm.graph)
    gm.recompile()

    assert not changed # not changed means count_calls before == count_calls after
    assert torch.allclose(gm(x), expected)


class FanOut(torch.nn.Module):
    """`h` has two users, one of them is dead, the other must stay."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.relu()
        dead = h.sin()  # noqa: F841
        return h + 1


def test_fan_out_keeps_shared_node():
    model = FanOut().eval()
    gm = fx.symbolic_trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x)
    before = count_calls(gm)

    changed = dce(gm.graph)
    gm.recompile()
    after = count_calls(gm)

    assert changed
    assert before - after == 1
    assert torch.allclose(gm(x), expected)

class NoDeadWork(torch.nn.Module):
    """The chain is complete and no dead-code found, `changed` should be `False`."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1
        z = y * 2
        return z

def test_no_dead_work_not_changed():
    model = NoDeadWork().eval()
    gm = fx.symbolic_trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x)

    changed = dce(gm.graph)
    gm.recompile()

    assert not changed
    assert torch.allclose(gm(x), expected)

class UnusedInput(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x

def test_unused_input_survives():
    model = UnusedInput().eval()
    gm = fx.symbolic_trace(model)
    x = torch.randn(INPUT_SIZE)
    y = torch.randn(INPUT_SIZE)
    expected = model(x, y)

    changed = dce(gm.graph)
    gm.recompile()

    assert not changed
    assert torch.allclose(gm(x, y), expected)

class ImpureFunction(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.Tensor([1, 2, 3])
        torch._assert(isinstance(y, torch.Tensor), f"y must be a torch.Tensor, found: {type(y)}")
        return x

def test_impure_function_survives():
    model = ImpureFunction().eval()
    gm = fx.symbolic_trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x)

    changed = dce(gm.graph)
    gm.recompile()

    assert not changed
    assert torch.allclose(gm(x), expected)