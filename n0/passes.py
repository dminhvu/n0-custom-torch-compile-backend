"""Constant folding, CSE, DCE and the fixpoint driver. You write these — PLAN.md § Spec.

Contract the grader relies on (`uv run python -m grader`):
- every pass mutates the graph in place and returns True iff it changed anything;
- `optimize` runs fold_constants → cse → dce until nothing changes, recompiles `gm`, returns it.
Signatures are the contract; everything else (helpers, names, structure) is yours.
"""

import torch.fx as fx

# Largest tensor fold_constants may produce, in elements (numel). Justify the value here.
FOLD_SIZE_CAP: int = 0  # TODO


def dce(graph: fx.Graph) -> bool:
    node = graph.output_node()
    while node and not node.op == 'placeholder':
        if not node.users and node.op.startswith("call_"):
            graph.erase_node(node)
        node = node.prev


def cse(graph: fx.Graph) -> bool:
    pass


def fold_constants(gm: fx.GraphModule) -> bool:
    pass


def optimize(gm: fx.GraphModule) -> fx.GraphModule:
    fold_constants(gm)
    cse(gm)
    dce(gm)
    return gm
