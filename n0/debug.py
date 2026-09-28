import torch
import torch.fx as fx

class DebugChain(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        c = torch.ones(3) * 5 + 1
        d = c.clamp(-5, 5)
        return x * d

chain = DebugChain()

trace = fx.symbolic_trace(chain)

print(trace.graph)