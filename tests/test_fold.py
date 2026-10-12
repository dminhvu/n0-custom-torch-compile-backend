import torch
import torch.fx as fx

from n0.passes import fold_constants

INPUT_SIZE = 100

class CallFunction(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = 2 * 3
        return x + y


def test_call_function_folded():
    model = CallFunction().eval()
    gm = fx.symbolic_trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x) 

    changed = fold_constants(gm)
    gm.recompile()

    print(gm.graph)

    assert changed
    assert torch.allclose(gm(x), expected)

class StatefulOp(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.randn(INPUT_SIZE)
        return x + y

def test_stateful_op_not_folded():
    model = StatefulOp().eval()
    gm = fx.symbolic_trace(model)
    x = torch.randn(INPUT_SIZE)
    _ = model(x) 

    changed = fold_constants(gm)
    gm.recompile()

    assert not changed
    # no output vs expected check since intermediate values are randomized
    # assert torch.allclose(gm(x), expected)