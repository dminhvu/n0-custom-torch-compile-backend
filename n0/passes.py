"""Constant folding, CSE, DCE and the fixpoint driver. You write these — PLAN.md § Spec.

Contract the grader relies on (`uv run python -m grader`):
- every pass mutates the graph in place and returns True iff it changed anything;
- `optimize` runs fold_constants → cse → dce until nothing changes, recompiles `gm`, returns it.
Signatures are the contract; everything else (helpers, names, structure) is yours.
"""


import torch.fx as fx
import torch.fx.node

# Largest tensor fold_constants may produce, in elements (numel). Justify the value here.
FOLD_SIZE_CAP: int = 1_000_000  # max 10^6 elements per Tensor: max 8 bytes/element (float64/int64) * 10^6 elements ~ 7.6 MB
STATEFUL_OPS = ["rand", "randn", "randint", "randperm", "rand_like", "bernoulli", "dropout"]

def _get_target_name(target) -> str:
    if isinstance(target, str):
        return target

    return getattr(target, "__name__", str(target))

def _is_side_effecting_node(node: fx.Node) -> bool:
    op_name = _get_target_name(node.target)

    if op_name.endswith("_") and not op_name.endswith("__"):
        return True
    if hasattr(node, "is_impure") and node.is_impure():
        return True
    return False

def dce(graph: fx.Graph) -> bool:
    """
    Perform dead-code elimination.
    A node is considered dead if it is not a side-effecting node and it doesn't have any user.
    """
    changed = False
    node: fx.Node
    for node in reversed(list(graph.nodes)):
        if _is_side_effecting_node(node):
            continue
        if len(node.users) == 0:
            graph.erase_node(node)
            changed = True            

    return changed


def cse(graph: fx.Graph) -> bool:
    raise NotImplementedError


def fold_constants(gm: fx.GraphModule) -> bool:
    changed = False
    env = {}
    node: fx.Node

    def resolve_arg(arg):
        if isinstance(arg, fx.Node):
            return env.get(arg), arg in env
        elif isinstance(arg, (tuple, list)):
            resolved = [resolve_arg(a) for a in arg]
            vals = [r[0] for r in resolved]
            all_const = all(r[1] for r in resolved)
            return type(arg)(vals), all_const
        elif isinstance(arg, dict):
            resolved = {k: resolve_arg(v) for k, v in arg.items()}
            vals = {k: v[0] for k, v in resolved.items()}
            all_const = all(v[1] for v in resolved.values())
            return vals, all_const
        else:
            return arg, True
    
    for node in list(gm.graph.nodes):
        # 'placeholder' and 'output' nodes are not candidates to be folded
        if node.op in ["placeholder", "output"] or _is_side_effecting_node(node):
            continue

        op_name = _get_target_name(node.target)
        if op_name in STATEFUL_OPS:
            continue

        target = node.target
        
        if node.op == "get_attr": # already a constant node, only retrieve the actual tensor
            result = getattr(gm, target)
            if isinstance(result, torch.Tensor) and result.numel() > FOLD_SIZE_CAP:
                continue
            env[node] = result
        elif node.op in ["call_function", "call_method"]: # candidates to be folded
            if any(_is_side_effecting_node(user) for user in list(node.users)):
                continue     

            args, args_const = resolve_arg(node.args)
            kwargs, kwargs_const = resolve_arg(node.kwargs)

            if not (args_const and kwargs_const):
                continue

            if node.op == "call_function":           
                result = target(*args, **kwargs)
            elif node.op == "call_method":
                obj, *args = args
                result = getattr(obj, target)(*args, **kwargs)

            if isinstance(result, torch.Tensor) and result.numel() > FOLD_SIZE_CAP:
                continue
            
            with gm.graph.inserting_before(node):
                if isinstance(result, torch.Tensor):
                    name = f"_folded_{str(node)}"
                    gm.register_buffer(name, result)
                    new_node = gm.graph.get_attr(name)
                    env[new_node] = result
                else:
                    new_node = result

            if isinstance(new_node, fx.Node):
                node.replace_all_uses_with(new_node)
            else:
                for user in list(node.users):
                    user.replace_input_with(node, new_node)

            env[node] = result
            gm.graph.erase_node(node)
            changed = True

    return changed


def optimize(gm: fx.GraphModule) -> fx.GraphModule:
    while True:
        fc_changed = fold_constants(gm)
        # cse(gm.graph)
        dce_changed = dce(gm.graph)
        gm.recompile()
        if not (fc_changed or dce_changed):
            break
    return gm
