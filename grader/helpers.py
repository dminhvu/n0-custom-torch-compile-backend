"""Graph capture and inspection helpers shared by the checks."""

import copy
from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx
from torch.utils._pytree import tree_leaves


def capture(fn: Callable, *inputs: Any) -> tuple[fx.GraphModule, list]:
    """The FX graph Dynamo hands to a backend for `fn(*inputs)`, plus its example_inputs."""
    torch._dynamo.reset()
    box: dict = {}

    def spy(gm: fx.GraphModule, example_inputs: list) -> Callable:
        box["gm"], box["ex"] = gm, list(example_inputs)
        return gm.forward

    torch.compile(fn, backend=spy, fullgraph=True)(*inputs)
    torch._dynamo.reset()
    return box["gm"], box["ex"]


def trace(module: torch.nn.Module) -> fx.GraphModule:
    """symbolic_trace: parameters/buffers appear as get_attr (Dynamo lifts them to placeholders)."""
    return fx.symbolic_trace(module)


def build(case: Any) -> tuple[fx.GraphModule, list, list[torch.Tensor]]:
    """(graph module as the case's tracer produces it, its inputs, eager outputs)."""
    module = case.make().eval()
    inputs = case.inputs()
    eager = [t for t in tree_leaves(module(*[i.clone() for i in inputs])) if isinstance(t, torch.Tensor)]
    if case.tracer == "dynamo":
        gm, ex = capture(module, *inputs)
        return gm, ex, eager
    return trace(module), inputs, eager


def run(gm: fx.GraphModule, inputs: list) -> list[torch.Tensor]:
    """Run gm on clones of the inputs (so in-place ops can't leak between runs); flat outputs."""
    args = [a.clone() if isinstance(a, torch.Tensor) else a for a in inputs]
    return [t for t in tree_leaves(gm(*args)) if isinstance(t, torch.Tensor)]


def snapshot(gm: fx.GraphModule) -> fx.GraphModule:
    return copy.deepcopy(gm)


def matches(target: Any, name: Any) -> bool:
    """Identity/equality, or a string matching a call_method target or a function's __name__."""
    if target is name or target == name:
        return True
    return isinstance(name, str) and getattr(target, "__name__", None) == name


def calls(gm: fx.GraphModule, target: Any = None, live_only: bool = False) -> list[fx.Node]:
    """call_* nodes (optionally with a given target). live_only: ignore nodes nobody uses —
    lets the fold/CSE checks count results without depending on your DCE."""
    out = []
    for n in gm.graph.nodes:
        if not n.op.startswith("call_"):
            continue
        if target is not None and not matches(n.target, target):
            continue
        if live_only and not n.users:
            continue
        out.append(n)
    return out


def get_attrs(gm: fx.GraphModule, live_only: bool = True) -> list[fx.Node]:
    return [n for n in gm.graph.nodes if n.op == "get_attr" and (n.users or not live_only)]


def fmt(nodes: list[fx.Node]) -> str:
    return ", ".join(f"%{n.name}={n.op}[{getattr(n.target, '__name__', n.target)}]" for n in nodes)


def assert_close(actual: list, expected: list, what: str) -> None:
    assert len(actual) == len(expected), f"{what}: {len(actual)} outputs, eager has {len(expected)}"
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a.shape == e.shape, f"{what}: output {i} has shape {tuple(a.shape)}, eager {tuple(e.shape)}"
        assert a.dtype == e.dtype, f"{what}: output {i} has dtype {a.dtype}, eager {e.dtype}"
        assert torch.allclose(a, e, equal_nan=True), f"{what}: output {i} differs from eager"
