# n0 — a custom `torch.compile` backend

This toy project implements a custom backend for `torch.compile` at the computation graph optimization level, with three passes: constant folding, dead-code elimination, and common subexpression elimination.

The project is called `n0` since it belongs to a series of projects that I'm trying to walk my first steps into the world of compiler engineering. There are other projects in the series, including `n1` (Elementwise fusion pass + Triton codegen), `n2` (LLVM/MLIR proper), and `n3` (Schedule search / autotuner) that will be updated later.

The main programming language used in this project is Python with PyTorch library, especially the PyTorch FX module, which is used to manipulate and interact with the computation graph.

## Optimization Passes

The backend supports three optimization passes in the following order:

### 1. Constant Folding

The first pass `fold_constants()` tries to find and evaluate constant expressions in the computation graph and replace them with the computed values. This helps reduce the number of actual computations to be performed during runtime.

Example: The expression `y = torch.add(2, 3)` can be replaced with `y = 5`.

However, it is not always beneficial to perform constant folding. When the number of elements in the tensor is too large, it may lead to increased memory usage and slower performance. Therefore, the pass includes a threshold `FOLD_SIZE_CAP = 1_000_000` to limit the size of tensors to be folded. This makes the maximum size in MB of the folded tensor to be around 7.6 MB (assuming an 8-byte data type like float64 or int64).

The pass also skips two kinds of nodes. Random operations like `torch.rand` or `dropout` are not folded, because their result must be different on every call. A value that is later changed by an in-place operation (for example `y.add_(1)`) is not folded either, because folding assumes the value is the same every time.

### 2. Common Subexpression Elimination

The second pass `cse()` finds nodes that compute the same thing twice and keeps only the first one. Two nodes are the same when they have the same operation and the same inputs, so the pass builds a key from `(op, target, args, kwargs)` for each node. If the key is already seen, all users of the node are moved to the first node.

Example: In `a = (x * 2).relu()` and `b = (x * 2).relu()`, the second `x * 2` and the second `relu` are replaced by the first ones.

Only pure nodes are merged. A random operation called twice gives two different results, so `torch.rand(2)` twice is kept as two nodes.

### 3. Dead-Code Elimination

The third pass `dce()` removes nodes whose result is never used. It walks the graph from the end to the start, so when a node is removed, the nodes it used can also be removed in the same walk. This works because the FX graph is one basic block in SSA form, and `node.users` gives the users of each value directly.

However, a node with no users is not always dead. An in-place operation like `y.add_(1)` changes `y` and returns nothing that is used, so it is kept. Nodes marked as impure by PyTorch are kept too.

### Running the passes to a fixpoint

`optimize()` runs fold → CSE → DCE in a loop until no pass changes the graph anymore, and then calls `gm.recompile()` to regenerate the Python code of the graph.

## How to run

The project uses [uv](https://docs.astral.sh/uv/), Python 3.12 and CPU-only PyTorch.

```sh
uv sync
uv run pytest
```

To use the backend, import `n0.backend` (this registers it under the name `"n0"`) and pass the name to `torch.compile`:

```python
import torch
import n0.backend  # registers "n0"

class Demo(torch.nn.Module):
    def forward(self, x):
        scale = torch.ones(3) * 2 + 1   # constant: folded
        a = (x * scale).relu()
        b = (x * scale).relu()          # same as a: merged by CSE
        unused = x.sin()                # never used: removed by DCE
        return a + b

model = Demo()
x = torch.randn(3)
compiled = torch.compile(model, backend="n0")
print(torch.allclose(compiled(x), model(x)))  # True
```

The backend prints the graph before and after the passes:

```
# before
graph():
    %l_x_ : torch.Tensor [num_users=3] = placeholder[target=L_x_]
    %ones : [num_users=1] = call_function[target=torch.ones](args = (3,), kwargs = {})
    %mul : [num_users=1] = call_function[target=operator.mul](args = (%ones, 2), kwargs = {})
    %scale : [num_users=2] = call_function[target=operator.add](args = (%mul, 1), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=operator.mul](args = (%l_x_, %scale), kwargs = {})
    %a : [num_users=1] = call_method[target=relu](args = (%mul_1,), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=operator.mul](args = (%l_x_, %scale), kwargs = {})
    %b : [num_users=1] = call_method[target=relu](args = (%mul_2,), kwargs = {})
    %unused : [num_users=0] = call_method[target=sin](args = (%l_x_,), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=operator.add](args = (%a, %b), kwargs = {})
    return (add_1,)

# after
graph():
    %l_x_ : torch.Tensor [num_users=1] = placeholder[target=L_x_]
    %_folded_scale : [num_users=1] = get_attr[target=_folded_scale]
    %mul_1 : [num_users=1] = call_function[target=operator.mul](args = (%l_x_, %_folded_scale), kwargs = {})
    %a : [num_users=1] = call_method[target=relu](args = (%mul_1,), kwargs = {})
    %add_1 : [num_users=1] = call_function[target=operator.add](args = (%a, %a), kwargs = {})
    return (add_1,)
```

| Node type | Before | After |
|---|---|---|
| `call_function` | 6 | 2 |
| `call_method` | 3 | 1 |
| `get_attr` | 0 | 1 |
| total call nodes | 9 | 3 |

Constant folding turned `torch.ones(3) * 2 + 1` into one buffer, CSE merged the second `x * scale` and `relu`, and DCE removed `x.sin()`.

## Freezing caveat

Folding a subexpression that only uses model parameters bakes the current weights into the graph. This is only correct for inference, when the weights don't change anymore (Inductor calls this *freezing*). Under `torch.compile`, Dynamo passes parameters into the graph as inputs (placeholders), so they are not folded there. They are only folded when the graph comes from `torch.fx.symbolic_trace`, where parameters are read with `get_attr`.

## Known issues

- CSE can merge two reads of a tensor across a single-argument in-place operation like `y.relu_()`, because only in-place calls with more than one argument clear the seen keys. Example: `y.sin(); y.relu_(); y.sin()` gives a different result from eager PyTorch.
- CSE also doesn't see writes through a view: after `v = y.view(-1); v.add_(1)`, a later `y.sin()` can still be merged with an earlier one.
- DCE removes a call that writes into an `out=` argument (`torch.add(x, 1, out=y)`), since it has no users and no `_` suffix.
