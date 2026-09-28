"""DCE unit tests: trace a small module with known dead work, run `dce` on its graph,
check the node count dropped by exactly that much and the output is unchanged."""

import torch
import torch.fx as fx

import pytest

from n0.passes import dce


def count_calls(gm: fx.GraphModule) -> int:
    """Number of call_* nodes — the ones DCE is allowed to remove."""
    return sum(n.op.startswith("call_") for n in gm.graph.nodes)


# --- Worked example ----------------------------------------------------------


class DeadChain(torch.nn.Module):
    """3 call nodes are dead (sin → cos → mul, result unused), 1 is live (add)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dead = x.sin().cos() * 2  # noqa: F841 — the point of the test
        return x + 1


def test_dead_chain_removed():
    # 1. Build the graph. symbolic_trace records every op, dead or not.
    model = DeadChain().eval()
    gm = fx.symbolic_trace(model)
    x = torch.randn(4)
    expected = model(x)  # eager result, before any pass touches the graph
    before = count_calls(gm)

    # 2. Run the pass on the graph, then regenerate gm.forward from it.
    changed = dce(gm.graph)
    gm.recompile()

    # 3. Assert: it reports a change, removed exactly the known dead work,
    #    is idempotent, agrees with torch's reference, and the result is unchanged.
    assert changed
    assert before - count_calls(gm) == 3
    assert not dce(gm.graph)
    assert not gm.graph.eliminate_dead_code()
    assert torch.allclose(gm(x), expected)


# --- Your turn: modules are ready, write the assertions ----------------------


class InplaceNoUsers(torch.nn.Module):
    """`y.add_(1)` has no users but mutates y, which is returned. Must survive."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2
        y.add_(1)
        return y


def test_inplace_survives():
    pytest.skip("TODO: which count do you expect, and why must allclose still hold?")


class FanOut(torch.nn.Module):
    """`h` has two users; only one of them is dead. h must stay, the dead user must go."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.relu()
        dead = h.sin()  # noqa: F841
        return h + 1


def test_fan_out_keeps_shared_node():
    pytest.skip("TODO: how many nodes are removed? Check `h` specifically, not just the count.")
