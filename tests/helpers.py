import torch
import torch.fx as fx

INPUT_SIZE = 100

def count_calls(gm: fx.GraphModule) -> int:
    """Number of call_* nodes: the ones DCE is allowed to remove."""
    return sum(n.op.startswith("call_") for n in list(gm.graph.nodes))

def trace(model: torch.nn.Module) -> fx.GraphModule:
    """Like fx.symbolic_trace, but buffers become get_attr nodes instead of baked-in constants."""
    tracer = fx.Tracer()
    tracer.proxy_buffer_attributes = True
    graph = tracer.trace(model)
    return fx.GraphModule(model, graph)

def get_target_name(target: torch.fx.node.Target) -> str:
    if isinstance(target, str):
        return target

    return getattr(target, "__name__", str(target))