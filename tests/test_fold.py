import torch
import torch.fx as fx

from n0.passes import fold_constants
from tests.helpers import INPUT_SIZE, trace, get_target_name


class CallFunction(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("c", torch.ones(INPUT_SIZE) * 5 + 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.c * 2 + 1)


def test_call_function_folded():
    model = CallFunction().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x)

    print("before", gm.graph)
     
    changed = fold_constants(gm)
    gm.recompile()

    print("after", gm.graph)

    assert changed
    assert torch.allclose(gm(x), expected)
    targets = [get_target_name(n.target) for n in list(gm.graph.nodes)]
    assert targets.count("_folded_mul") == 1
    assert targets.count("_folded_add") == 1

    # Idempotency check
    changed_second = fold_constants(gm)

    assert not changed_second

class CallMethod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("c", torch.ones(INPUT_SIZE) * 5 + 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c.mul(2).add(1)

def test_call_method_folded():
    model = CallMethod().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x)

    print("before", gm.graph)
    
    changed = fold_constants(gm)
    gm.recompile()

    assert changed
    assert torch.allclose(gm(x), expected)

    print("after", gm.graph)

    targets = [get_target_name(n.target) for n in list(gm.graph.nodes)]
    assert targets.count("_folded_mul") == 1
    assert targets.count("_folded_add") == 1

    # Idempotency check
    changed_second = fold_constants(gm)

    assert not changed_second
class StatefulOp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("c", torch.ones(INPUT_SIZE) * 5 + 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.dropout(self.c, p=0.5, train=True)

def test_stateful_op_not_folded():
    model = StatefulOp().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    _ = model(x)
    
    changed = fold_constants(gm)
    gm.recompile()

    assert not changed
    targets = [get_target_name(n.target) for n in list(gm.graph.nodes)]
    print(gm.graph)
    print(targets)
    assert targets.count("dropout") == 1

    # Idempotency check
    changed_second = fold_constants(gm)

    assert not changed_second

class BelowFoldCapSize(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("c", torch.ones(1_000_000) * 5 + 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c * 2

def test_below_fold_cap_size_folded():
    model = BelowFoldCapSize().eval()
    gm = trace(model)
    x = torch.randn(1_000_000)
    expected = model(x)
    
    changed = fold_constants(gm)
    gm.recompile()

    assert changed
    assert torch.allclose(gm(x), expected)
    targets = [get_target_name(n.target) for n in list(gm.graph.nodes)]
    assert targets.count("_folded_mul") == 1

    # Idempotency check
    changed_second = fold_constants(gm)

    assert not changed_second

class AboveFoldCapSize(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("c", torch.ones(1_000_001) * 5 + 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.c * 2
        return y

def test_above_fold_cap_size_not_folded():
    model = AboveFoldCapSize().eval()
    gm = trace(model)
    x = torch.randn(1_000_001)
    _ = model(x)

    changed = fold_constants(gm)
    gm.recompile()

    assert not changed
    targets = [get_target_name(n.target) for n in list(gm.graph.nodes)]
    assert targets.count("_folded_mul") == 0

    # Idempotency check
    changed_second = fold_constants(gm)

    assert not changed_second

class InplaceMethod(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("c", torch.ones(INPUT_SIZE) * 5 + 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c * 2
        y.add_(1)
        return x + y

def test_inplace_method_not_folded():
    model = InplaceMethod().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    _ = model(x)
    
    changed = fold_constants(gm)
    gm.recompile()

    assert not changed

    # Idempotency check
    changed_second = fold_constants(gm)

    assert not changed_second