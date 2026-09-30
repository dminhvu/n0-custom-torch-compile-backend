"""Constant folding, CSE, DCE and the fixpoint driver. You write these — PLAN.md § Spec.

Contract the grader relies on (`uv run python -m grader`):
- every pass mutates the graph in place and returns True iff it changed anything;
- `optimize` runs fold_constants → cse → dce until nothing changes, recompiles `gm`, returns it.
Signatures are the contract; everything else (helpers, names, structure) is yours.
"""


import torch.fx as fx
import torch.fx.node

# Largest tensor fold_constants may produce, in elements (numel). Justify the value here.
FOLD_SIZE_CAP: int = 1e6  # TODO
NON_FOLDABLE_METHODS = ["rand", "randn", "randint", "randperm", "randperm", "rand_like", "bernoulli", "dropout"]

def _is_side_effecting_node(node: fx.Node) -> bool:
    op_name = node.target if isinstance(node.target, str) else node.target.__name__
    
    return True if op_name.endswith("_") or node.is_impure() else False

def _is_foldable_node(node: fx.Node) -> bool:
    if _is_side_effecting_node(node):
        return False
    op_name = node.target if isinstance(node.target, str) else node.target.__name__
    if op_name in NON_FOLDABLE_METHODS:
        return False
    input_nodes = node.all_input_nodes
    for node in input_nodes:
        if not node.op == 'get_attr':
            return False

    return True

def dce(graph: fx.Graph) -> bool:
    dead_code_found = False
    node: fx.Node
    for node in reversed(graph.nodes):
        if not _is_side_effecting_node(node) and not node.users:
            graph.erase_node(node)
            dead_code_found = True            

    return dead_code_found


def cse(graph: fx.Graph) -> bool:
    raise NotImplementedError



def fold_constants(gm: fx.GraphModule) -> bool:
    foldable_node_found = False
    d = {}
    node: fx.Node
    def map_nodes_to_values(nodes):
        return torch.fx.map_arg(nodes, lambda node: d[node])
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            paths = node.target.split(".")
            result = gm
            for path in paths:
                result = getattr(result, path)
            d[node] = result
        elif _is_foldable_node(node):
            if node.op == "call_function":
                target: torch.fx.node.Target = node.target
                args = map_nodes_to_values(node.args)
                kwargs = map_nodes_to_values(node.kwargs)
                result = target(*args, **kwargs)
                d[node] = result
            elif node.op == "call_method":
                target: str = node.target
                obj, *args = map_nodes_to_values(node.args)
                kwargs = map_nodes_to_values(node.kwargs)
                result = getattr(obj, target)(*args, **kwargs)
                d[node] = result
            foldable_node_found = True
    for node in gm.graph.nodes:
        if node in d:
            gm.register_buffer("_" + str(node), d[node])
            with gm.graph.inserting_before(node):
                new_node = gm.graph.get_attr("_" + str(node))
                node.replace_all_uses_with(new_node)
                d[new_node] = d[node]
    return foldable_node_found


def optimize(gm: fx.GraphModule) -> fx.GraphModule:
    raise NotImplementedError
    fold_constants(gm)
    # cse(gm.graph)
    dce(gm.graph)
    return gm
