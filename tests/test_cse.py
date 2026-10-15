import torch

from n0.passes import cse
from tests.helpers import INPUT_SIZE, get_target_name, trace


class MergeableNodes(torch.nn.Module):
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + 2
        x.add_(5)
        b = x + 2
        return a * b


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
    assert len(add_node.users) == 1
    assert get_target_name(add_1_node) == "add_1"
    assert len(add_1_node.users) == 1

    # Idempotency check
    changed_second = cse(gm.graph)

    assert not changed_second


class ImpureNodesNoArgsBetween(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + 2
        x.neg_()
        b = x + 2
        return a * b


def test_impure_nodes_no_args_between_not_merged():
    model = ImpureNodesNoArgsBetween().eval()
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
    assert len(add_node.users) == 1
    assert get_target_name(add_1_node) == "add_1"
    assert len(add_1_node.users) == 1

    # Idempotency check
    changed_second = cse(gm.graph)

    assert not changed_second

class ViewNode(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x + 1
        x.view(-1)[0:4].add_(5)
        b = x + 1
        return a + b


def test_view_node_not_merged():
    model = ViewNode().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x.clone())

    print("before", gm.graph)
     
    changed = cse(gm.graph)
    gm.recompile()

    print("after", gm.graph)

    assert not changed
    assert torch.allclose(gm(x.clone()), expected)
    filtered_nodes = list(filter(
        lambda n: n.name in ("add", "add_1") and n.op == "call_function",
        list(gm.graph.nodes)
    ))
    assert len(filtered_nodes) == 2
    add_node, add_1_node = filtered_nodes
    print(gm.graph)
    assert get_target_name(add_node) == "add"
    assert len(add_node.users) == 1
    assert get_target_name(add_1_node) == "add_1"
    assert len(add_1_node.users) == 1

    # Idempotency check
    changed_second = cse(gm.graph)

    assert not changed_second

class ViewWritesToBaseNode(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.view(-1)
        a = v + 1
        x.add_(5)
        b = v + 1
        return a + b


def test_view_writes_to_base_node_not_merged():
    model = ViewWritesToBaseNode().eval()
    gm = trace(model)
    x = torch.randn(INPUT_SIZE)
    expected = model(x.clone())

    print("before", gm.graph)
     
    changed = cse(gm.graph)
    gm.recompile()

    print("after", gm.graph)

    assert not changed
    assert torch.allclose(gm(x.clone()), expected)
    filtered_nodes = list(filter(
        lambda n: n.name in ("add", "add_1") and n.op == "call_function",
        list(gm.graph.nodes)
    ))
    assert len(filtered_nodes) == 2
    add_node, add_1_node = filtered_nodes
    print(gm.graph)
    assert get_target_name(add_node) == "add"
    assert len(add_node.users) == 1
    assert get_target_name(add_1_node) == "add_1"
    assert len(add_1_node.users) == 1

    # Idempotency check
    changed_second = cse(gm.graph)

    assert not changed_second