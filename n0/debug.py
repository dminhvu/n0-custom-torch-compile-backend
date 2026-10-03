import torch
import torch.fx as fx

class DebugChain(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = x.sum()
        a = x.add_(1)
        s2 = x.sum()
        return s1 + a + s2

chain = DebugChain()

trace = fx.symbolic_trace(chain)

# print(trace.graph)
# node: fx.Node
for node in trace.graph.nodes:
    print(node, node.op, node.target, node.args, node.all_input_nodes, sep=" | ")
