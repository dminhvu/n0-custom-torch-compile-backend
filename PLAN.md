# N0 — custom `torch.compile` backend

Started 2026-09-18 · Deadline **Wed 30 Sep 2026** — extended on 27 Sep from Sun 27 Sep, once
(hard: N1 starts in October, lectures 13 Oct)
Budget: **one weekend of code + one evening of write-up.** If it runs past two weekends, cut scope, don't extend.

Roadmap context: `~/.claude/career-plan.md` → "Project roadmap". N0 is the first of N0–N3 and
the cheapest artifact that turns "compiler development experience: none" into "some".

## Goal

A backend for `torch.compile` that receives the FX graph Dynamo captured, runs three classic
optimizer passes written from scratch — **constant folding, dead-code elimination, common
subexpression elimination** — and prints the graph before/after with node counts. ~200 lines.

Not a graph printer: the three passes are what make it an optimizer (career plan, deck 01 p.30).

## How this is worked

- **You write every line of the passes.** Claude reviews, explains, points at docs and at the
  CC lecture material, and writes at most test scaffolding you ask for. Reason: the CV test is
  "can you answer *how?* and *why that way?* twice deep" — that only holds for code you wrote.
- Commit after each milestone with a message that says what the pass does and one thing you learned.
- Log each session at the bottom of this file. Two lines is enough.
- **Autograder:** `uv run python -m grader` scores A1–C2 (70 of 100 points) after every change,
  with a hint per failed check; `uv run python -m grader dce` runs only matching checks. C3 and D
  (30) stay manual. The grader is Claude's; `tests/` is yours (it's graded in C2).

## Spec

```
backend(gm: torch.fx.GraphModule, example_inputs: list[Tensor]) -> Callable
```

1. Register with `torch._dynamo.register_backend` (or pass the callable to `torch.compile(backend=...)`).
2. Dump `gm.graph` before (node count per `op` type).
3. Run passes to a fixpoint: `fold_constants` → `cse` → `dce`, repeat until the graph stops changing.
4. `gm.recompile()`, dump after, return `gm.forward`.
5. Test modules (`tests/`), each with a *known* amount of removable work, asserting
   `torch.allclose(compiled(x), eager(x))` **and** the expected node-count drop.

### Pass requirements — the "why that way" list

| Pass | Must | Must not |
|---|---|---|
| **DCE** | Walk nodes in reverse; erase `call_*` nodes with no users. Treat the FX graph as one SSA basic block (`node.users` = def-use chain, CC deck on SSA). | Delete side-effecting nodes: in-place ops (`add_`, `copy_`, …), `placeholder`, `output`. Check `torch.fx.node._side_effectful_functions` and the `_` suffix convention. |
| **Constant folding** | Fold a node when every tensor input is a compile-time constant (`get_attr` of a buffer/param or a Python literal). Evaluate once, `register_buffer` the result, replace with `get_attr`. | Fold nondeterministic ops (`rand*`, `randn`, dropout in train mode), side-effecting ops, or anything producing a tensor bigger than a size cap (pick one, justify it). |
| **CSE** | Value-number each pure node by `(op, target, args-with-Node→id, kwargs)`; replace duplicates with `replace_all_uses_with`. | Merge impure nodes, or nodes whose args differ only in kwargs, dtype, device. Merge nodes that read memory an in-place op wrote to in between. List args must compare by value (check what type FX actually stores them as). |

**Known trade-off to document, not hide:** folding over parameters bakes weights in — it is
only valid for inference (this is what Inductor calls *freezing*). Say so in the README.

**Compare against the reference:** after your DCE, run `gm.graph.eliminate_dead_code()` and
assert it finds nothing left. Same idea with `torch.fx.passes` for CSE if it exists in your version.

**Verified on torch 2.14 (2026-09-23):**
- `eliminate_dead_code()` erases an unused `y.add_(1)` (`call_method` in-place ops report
  `is_impure() == False`), which changes the result. It assumes functional graphs; yours are
  not. So "finds nothing left" means *nothing except in-place nodes*. Worth a paragraph in the blog post.
- Dynamo always inlines `nn.Module`s and lifts parameters/buffers to **placeholders**, so
  parameter-only subexpressions are not foldable in the graph your backend receives. They are
  `get_attr` only under `fx.symbolic_trace`. `torch.ones(3)` shows up as a node under Dynamo;
  `symbolic_trace` evaluates it at trace time.

## Milestones

- [x] **M0 · Setup** — `uv` project, Python 3.12, CPU torch, pytest, git init. *(2026-09-18)*
- [ ] **M1 · Identity backend** — registered, called by `torch.compile`, prints the graph, returns
      `gm.forward`. Confirm outputs match eager. Trigger a graph break on purpose and see the
      backend called twice; then use `fullgraph=True`.
- [ ] **M2 · DCE** — own implementation + test module with dead branches. Reference check passes.
- [ ] **M3 · Constant folding** — test module with a `torch.ones(...) * 2 + 1`-style subgraph and
      a parameter-only subexpression. Size cap in place.
- [ ] **M4 · CSE** — test module that computes the same subexpression twice. kwargs test.
- [ ] **M5 · Fixpoint driver + node-count report** — the three passes interact (folding creates
      dead nodes; CSE creates dead nodes). Show a case where a single pass ordering misses something.
- [ ] **M6 · Demo + README** — run on a small real model (e.g. a 2-layer MLP, a tiny ResNet block
      from `torchvision` or hand-written). Before/after tabular dump + counts in the README.
- [ ] **M7 · Blog post** on dminhvu.com — pipeline diagram, what each pass does, the freezing
      trade-off, numbers. One CV bullet drafted.

## Grading rubric — score it yourself at the end, honestly

100 points. Map: ≥ 90 → 1.0 · 80 → 1.7 · 70 → 2.3 · 60 → 3.0 · < 60 → redo the weak section.

### A · Correctness — 30
- [ ] 10 — every test module: compiled output `allclose` to eager, including a parameter-heavy one
- [ ] 10 — passes are **idempotent**: running the pipeline twice changes nothing on the second run
- [ ] 5 — an in-place op in the test graph survives DCE, and the result is still correct
- [ ] 5 — reference `eliminate_dead_code()` finds nothing after your DCE, on every test

### B · The passes — 30
- [ ] 10 — DCE: correct reverse walk, side-effect check, no special-casing by node name
- [ ] 10 — folding: constant detection is recursive over args, nondeterminism excluded, size cap
      justified in a comment, folded values become `get_attr`
- [ ] 10 — CSE: real value numbering (not string-compare of `repr(node)`), kwargs and list-args
      handled, only pure nodes merged

### C · Engineering — 15
- [ ] 5 — core ≤ ~300 lines, one function per pass over `fx.Graph`, type hints, no dead flags
- [ ] 5 — `pytest` green, one test file per pass, one integration test through `torch.compile`
- [ ] 5 — README: what it is, how to run, before/after dump, node counts, the freezing caveat;
      commit history reads as a sequence of steps

### D · Understanding — 25 · answer aloud, no notes, before calling it done
Score 0 / 1.5 / 2.5 per question — zero if you'd have to look it up.
1. Draw eager → Dynamo → FX → AOTAutograd → Inductor → Triton. Where does your backend sit, and
   why do you see `torch.*` ops rather than `aten.*`? What changes if you wrap it in `aot_autograd`?
2. What is a graph break? Why does it make Dynamo call your backend more than once? What are guards?
3. The FX graph is an SSA basic block. What does that make trivial about DCE compared to DCE on
   LLVM IR with a CFG? (Think: control flow, memory side effects, liveness.)
4. Why can't you fold `torch.rand`? Why can't you DCE `x.add_(1)` even with no users?
5. Why is folding a parameter-only subexpression a trade-off? Who breaks if you're wrong?
6. What makes CSE sound? Relate to value numbering / GVN from the CC lectures.
7. Why fold → CSE → DCE, and why to a fixpoint? Give an example a single pass in that order misses.
8. What does Inductor do that you didn't? (This is the N1 pitch.)
9. What does `gm.recompile()` actually regenerate? What is `gm.code`?
10. In what sense is this "compiler development experience", and where would a compiler engineer
    say it is thin? (Be honest — this is the interview answer.)

**Total: ___ / 100 → grade ___** &nbsp;·&nbsp; dated ______

## Scope guard — do not

- No Triton, no codegen, no fusion → that is N1.
- No `aten` decomposition / AOTAutograd unless M1–M5 are done and you have time left.
- No shape specialisation, no dynamic shapes, no training support.
- No third weekend.

## Deliverables

1. `n0/` repo, public on GitHub, README with numbers.
2. Blog post on dminhvu.com.
3. One CV bullet you can defend twice deep, e.g. *"Wrote a `torch.compile` backend implementing
   constant folding, DCE and CSE over FX graphs; −N % nodes on <model>."*

## Reading (only what's needed)

- `torch.fx` docs: `Graph`, `Node`, `GraphModule`, `Interpreter` — read the *Node* section first.
- `torch.compile` "custom backends" docs page — the register_backend part only.
- CC deck on SSA and def-use chains; deck 01 p.30 for the optimisation list.
- Look at `torch/fx/passes/` *after* M4, not before — compare, don't copy.

## Log

<!-- date · what got done · one thing learned / one thing stuck -->

- 2026-09-18 · M0: repo, uv, torch CPU, this plan. Start coding 19 or 22 Sep.
- 2026-09-23 · Autograder `grader/` (99 checks, rubric A1–C2 = 70 pts) + pass stubs with the
  contract. Validated: hidden reference 70/70, every planted bug (mutation test) loses points; reference deleted.
  Learned: torch's `eliminate_dead_code` erases in-place `call_method`s; Dynamo lifts params to placeholders.
- 2026-09-26 · M1/M2 started: backend registered as "n0", DCE reverse walk (grader 21.5/70). Stuck:
  `optimize` passes `gm` where `dce` takes a `Graph` (breaks all of A1), no bool returns, no fixpoint,
  no `recompile`, DCE has no side-effect check (in-place, `_assert`, `setitem` erased). No `tests/` yet.
- 2026-09-27/28 · DCE complete (B1 10/10, A3, A4 green); `tests/test_dce.py` = Claude's worked example +
  2 stubs for you (M2 ticks when those are filled). Deadline → Wed 30 Sep. Fold started: B2 3/10, total 24.7/70.
  Learned: reverse walk is enough because the node list is a topological order; Dynamo's `gm.training`
  is always True. Stuck: which nodes are foldable — value map vs. rewritten nodes, get_attr refolded, rand/mutated consts folded.
