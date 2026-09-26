"""M1 · the backend, through torch.compile."""

import pytest
import torch
import torch.fx as fx

from grader import modules as M
from grader.helpers import assert_close, calls, capture, fmt, run

pytestmark = pytest.mark.fixpoint


@pytest.mark.rubric("A1")
def test_registered_as_n0():
    """import n0.backend registers it: torch.compile(model, backend="n0") works"""
    import n0.backend  # noqa: F401

    module, x = M.MLP().eval(), torch.randn(2, 8)
    assert "n0" in torch._dynamo.list_backends(exclude_tags=()), (
        "no backend named 'n0' — see torch._dynamo.register_backend"
    )
    compiled = torch.compile(module, backend="n0", fullgraph=True)
    assert torch.allclose(compiled(x), module(x))


@pytest.mark.rubric("A1")
def test_returns_working_callable():
    """n0_backend(gm, example_inputs) returns a callable that matches eager"""
    from n0.backend import n0_backend

    module, x = M.MLP().eval(), torch.randn(2, 8)
    gm, ex = capture(module, x)
    fn = n0_backend(gm, ex)
    assert callable(fn), "the backend must return a callable (e.g. gm.forward)"
    assert_close(run(fn, ex), [module(x)], "MLP via n0_backend")


@pytest.mark.rubric("A1")
def test_optimizes_the_graph():
    """the graph Dynamo passes in actually gets optimized"""
    from n0.backend import n0_backend

    seen: list[fx.GraphModule] = []

    def spy(gm, example_inputs):
        seen.append(gm)
        return n0_backend(gm, example_inputs)

    module, x = M.MLP().eval(), torch.randn(2, 8)
    out = torch.compile(module, backend=spy, fullgraph=True)(x)
    assert torch.allclose(out, module(x))
    left = calls(seen[0])
    assert len(left) <= 6, f"backend returned without optimizing: {fmt(left)} — does it call optimize(gm)?"


def _with_break(x):
    a = (x * 2).sin()
    torch._dynamo.graph_break()
    return a + torch.ones(3) * 3


@pytest.mark.rubric("A1")
def test_graph_break():
    """a graph break → the backend is called once per subgraph, result still right"""
    from n0.backend import n0_backend

    count = 0

    def spy(gm, example_inputs):
        nonlocal count
        count += 1
        return n0_backend(gm, example_inputs)

    x = torch.randn(3)
    out = torch.compile(_with_break, backend=spy)(x)
    assert count == 2, f"backend called {count}×, expected 2 (one per side of the break)"
    assert torch.allclose(out, _with_break(x))
