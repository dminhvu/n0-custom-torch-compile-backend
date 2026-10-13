import torch
import torch.fx as fx

from n0.passes import cse
from tests.helpers import INPUT_SIZE, get_target_name, trace


class MergeableNodes(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = (x + 2).relu()
        b = (x + 2).relu()
        return a + b


def test_mergeable_nodes_merged():
    model = MergeableNodes().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x)

    print("before", gm.graph)
     
    changed = cse(gm.graph)
    gm.recompile()

    print("after", gm.graph)

    assert changed
    assert torch.allclose(gm(x), expected)
    filtered_nodes = list(filter(
        lambda n: n.name in ["add", "add_1"] and n.op == "call_function",
        list(gm.graph.nodes)
    ))
    assert len(filtered_nodes) == 2
    add_node, add_1_node = filtered_nodes
    print(gm.graph)
    assert get_target_name(add_node) == "add"
    assert len(add_node.users) == 2
    assert get_target_name(add_1_node) == "add_1"
    assert len(add_1_node.users) == 0

    # Idempotency check
    changed_second = cse(gm.graph)

    assert not changed_second


class StatefulOps(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.rand_like(x)
        b = torch.rand_like(x)
        return a + b


def test_stateful_ops_not_merged():
    model = StatefulOps().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    _ = model(x)

    print("before", gm.graph)
     
    changed = cse(gm.graph)
    gm.recompile()

    print("after", gm.graph)

    assert not changed
    filtered_nodes = list(filter(
        lambda n: n.name in ("rand_like", "rand_like_1"),
        list(gm.graph.nodes)
    ))
    assert len(filtered_nodes) == 2
    rand_like_node, rand_like_1_node = filtered_nodes
    print(gm.graph)
    assert get_target_name(rand_like_node) == "rand_like"
    assert len(rand_like_node.users) == 1
    assert get_target_name(rand_like_1_node) == "rand_like_1"
    assert len(rand_like_1_node.users) == 1

    # Idempotency check
    changed_second = cse(gm.graph)

    assert not changed_second

class ImpureNodesBetween(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 2
        x.add_(5)
        b = x + 2
        return x + b


def test_impure_nodes_between_not_merged():
    model = ImpureNodesBetween().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    _ = model(x)

    print("before", gm.graph)
     
    changed = cse(gm.graph)
    gm.recompile()

    print("after", gm.graph)

    assert not changed
    filtered_nodes = list(filter(
        lambda n: n.name in ("add", "add_1") and n.op == "call_function",
        list(gm.graph.nodes)
    ))
    assert len(filtered_nodes) == 2
    add_node, add_1_node = filtered_nodes
    print(gm.graph)
    assert get_target_name(add_node) == "add"
    assert len(add_node.users) == 3
    assert get_target_name(add_1_node) == "add_1"
    assert len(add_1_node.users) == 1

    # Idempotency check
    changed_second = cse(gm.graph)

    assert not changed_second


class ImpureNodesNoArgsBetween(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 2
        x.neg_()
        b = x + 2
        return x + b


def test_impure_nodes_no_args_between_not_merged():
    model = ImpureNodesBetween().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    _ = model(x)

    print("before", gm.graph)
     
    changed = cse(gm.graph)
    gm.recompile()

    print("after", gm.graph)

    assert not changed
    filtered_nodes = list(filter(
        lambda n: n.name in ("add", "add_1") and n.op == "call_function",
        list(gm.graph.nodes)
    ))
    assert len(filtered_nodes) == 2
    add_node, add_1_node = filtered_nodes
    print(gm.graph)
    assert get_target_name(add_node) == "add"
    assert len(add_node.users) == 3
    assert get_target_name(add_1_node) == "add_1"
    assert len(add_1_node.users) == 1

    # Idempotency check
    changed_second = cse(gm.graph)

    assert not changed_second