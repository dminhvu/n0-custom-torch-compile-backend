"""Test modules with a known amount of dead / constant / duplicate work.

Each Case says how its graph is produced (Dynamo, as your backend sees it, or symbolic_trace,
where parameters are get_attr) and how many call_* nodes must remain after `optimize`.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- dead code -----------------------------------------------------------------------------

class DeadChain(nn.Module):
    def forward(self, x):
        a = x + 1
        b = a * 2
        c = b - 3  # noqa: F841  (a -> b -> c is dead)
        return x.relu()


class DeadFeedsDead(nn.Module):
    def forward(self, x):
        y = x.sin()
        dead1 = y.cos()
        z = y * 2
        dead2 = (z + dead1).exp()  # noqa: F841  (dead1's only user is dead)
        return z


class InPlace(nn.Module):
    def forward(self, x):
        y = x.clone()
        y.add_(1)
        y.mul_(2)
        dead = x * 7  # noqa: F841
        return y


class SetItem(nn.Module):
    def forward(self, x):
        y = x.clone()
        y[0] = 5.0  # operator.setitem: in-place, no trailing underscore, zero users
        return y


class BufferCopy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("state", torch.zeros(3))

    def forward(self, x):
        self.state.copy_(x)
        return x * 2


class UnusedInput(nn.Module):
    def forward(self, x, unused):
        return x + 1


# --- constants -----------------------------------------------------------------------------

class ConstChain(nn.Module):
    def forward(self, x):
        return x + (torch.ones(3, 3) * 2 + 1)


class ParamOnly(nn.Module):
    """Inference-only: folding w.t() * 2 + b bakes the weights in (freezing)."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 4))
        self.register_buffer("b", torch.randn(4))

    def forward(self, x):
        return x @ (self.w.t() * 2 + self.b)


class ConstList(nn.Module):
    def forward(self, x):
        return x + torch.cat([torch.ones(2), torch.zeros(2)])


class MixedList(nn.Module):
    def forward(self, x):
        return torch.cat([x, torch.ones(2)])


class ShapeOfParam(nn.Module):
    """self.w.shape[0] is a constant too — but an int, not a tensor."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        return x.view(self.w.shape[0], -1) + self.w.sum()


class MutatedConst(nn.Module):
    """torch.zeros(3) looks constant, but add_ writes to it on every call."""

    def forward(self, x):
        t = torch.zeros(3)
        t.add_(1)
        return x + t


class BigConst(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x + torch.ones(self.n)


RANDOM_OPS: dict[str, tuple[Callable, Callable]] = {
    "torch.rand": (lambda x: x + torch.rand(3), torch.rand),
    "torch.randn": (lambda x: x + torch.randn(3), torch.randn),
    "torch.randint": (lambda x: x + torch.randint(0, 10, (3,)), torch.randint),
    "torch.randperm": (lambda x: x + torch.randperm(3), torch.randperm),
    "torch.rand_like": (lambda x: x + torch.rand_like(torch.ones(3)), torch.rand_like),
    "torch.bernoulli": (lambda x: x + torch.bernoulli(torch.full((3,), 0.5)), torch.bernoulli),
    "F.dropout(training=True)": (lambda x: x + F.dropout(torch.ones(3), 0.5, training=True), F.dropout),
}


# --- duplicates ----------------------------------------------------------------------------

class DupChain(nn.Module):
    def forward(self, x, y):
        a = (x * 3).sin()
        b = (x * 3).sin()
        c = torch.cat([x, y])
        d = torch.cat([x, y])
        return a + b, c + d


# --- a realistic one -----------------------------------------------------------------------

class MLP(nn.Module):
    """2-layer MLP (F.linear on own parameters, so both tracers see the same ops) plus a
    duplicated layer, a dead branch and a constant subexpression."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(16, 8))
        self.b1 = nn.Parameter(torch.randn(16))
        self.w2 = nn.Parameter(torch.randn(4, 16))
        self.b2 = nn.Parameter(torch.randn(4))
        self.register_buffer("scale", torch.full((4,), 0.5))

    def forward(self, x):
        h = F.relu(F.linear(x, self.w1, self.b1))
        h_again = F.relu(F.linear(x, self.w1, self.b1))
        dead = F.linear(h_again, self.w2) * 3  # noqa: F841
        out = F.linear(h, self.w2, self.b2) * (self.scale * 2)
        return out + (torch.ones(4) * 0.1 + 0.2)


@dataclass
class Case:
    name: str
    make: Callable[[], nn.Module]
    inputs: Callable[[], list]
    tracer: str  # "dynamo" | "fx"
    max_calls: int  # call_* nodes allowed after optimize (all remaining must be needed)
    random: bool = False  # output not comparable to eager
    mutates_state: bool = False
    tags: list[str] = field(default_factory=list)


def _x(*shape: int) -> Callable[[], list]:
    return lambda: [torch.randn(*shape)]


CASES = [
    Case("DeadChain", DeadChain, _x(3), "dynamo", max_calls=1),
    Case("DeadFeedsDead", DeadFeedsDead, _x(3), "dynamo", max_calls=2),
    Case("InPlace", InPlace, _x(3), "dynamo", max_calls=3),
    Case("SetItem", SetItem, _x(3), "dynamo", max_calls=2),
    Case("BufferCopy", BufferCopy, _x(3), "dynamo", max_calls=2, mutates_state=True),
    Case("ConstChain", ConstChain, _x(3, 3), "dynamo", max_calls=1),
    Case("ConstList", ConstList, _x(4), "dynamo", max_calls=1),
    Case("MixedList", MixedList, _x(2), "dynamo", max_calls=1),
    Case("MutatedConst", MutatedConst, _x(3), "dynamo", max_calls=3),
    Case("DupChain", DupChain, lambda: [torch.randn(2, 2), torch.randn(2, 2)], "dynamo", max_calls=5),
    Case("MLP[dynamo]", MLP, _x(2, 8), "dynamo", max_calls=6),
    Case("ParamOnly[fx]", ParamOnly, _x(3, 4), "fx", max_calls=1),
    Case("ShapeOfParam[fx]", ShapeOfParam, _x(4, 4), "fx", max_calls=4),
    Case("MLP[fx]", MLP, _x(2, 8), "fx", max_calls=5),
]
