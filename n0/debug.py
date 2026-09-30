import torch
import torch.fx as fx

class DebugChain(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        c = torch.rand(3) * 5
        d = c.clamp(-5, 5)
        return (x * d).clamp(-2, 2)

chain = DebugChain()

trace = fx.symbolic_trace(chain)

# print(trace.graph)
node: fx.Node
for node in trace.graph.nodes:
    print(node, node.op, node.target, node.args, node.all_input_nodes, sep=" | ")
