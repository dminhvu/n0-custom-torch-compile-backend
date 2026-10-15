import torch

from n0.backend import n0_backend  # noqa: F401
from tests.helpers import INPUT_SIZE


class Model(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.add_(5)
        y = x + 1
        x.neg_()
        v = x + 1
        dead = torch.ones(10)
        c = torch.ones(INPUT_SIZE) * 2 + 5
        return c + v + y

def test_model():
    x = torch.randn(INPUT_SIZE)
    eager_model = Model()
    expected = eager_model(x.clone())

    n0_model = torch.compile(Model(), backend="n0")

    assert torch.allclose(n0_model(x.clone()), expected)
    # second check intentionally
    assert torch.allclose(n0_model(x.clone()), expected)
