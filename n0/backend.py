"""The torch.compile backend. You write this — PLAN.md § Spec, milestone M1.

Contract the grader relies on:
- `n0_backend` optimizes `gm` in place (via `passes.optimize`) and returns a callable;
- importing this module registers the backend under the name "n0",
  so `torch.compile(model, backend="n0")` works.
"""

from collections.abc import Callable

import torch
import torch.fx as fx

from n0.passes import optimize


def n0_backend(gm: fx.GraphModule, example_inputs: list[torch.Tensor]) -> Callable:
    print(gm.graph)
    gm = optimize(gm)
    print(gm.graph)
    return gm.forward

torch._dynamo.register_backend(n0_backend, name="n0")