# n0 — portfolio project, not a coding task

Read `PLAN.md` first. It holds the spec, milestones, rubric and the session log.

## Rules for Claude in this repo

- **The user writes the optimizer passes** (`fold_constants`, `cse`, `dce`, the fixpoint driver,
  the backend registration). Do not write or rewrite them, even if asked casually — instead
  explain, review, point at the relevant `torch.fx` API or CC lecture concept, and ask the
  guiding question. If the user explicitly insists after being reminded, comply.
- Fine to write on request: pytest scaffolding, test *modules* (small `nn.Module`s with known
  dead/constant/duplicate work), the README skeleton, shell/uv commands.
- When reviewing, grade against the rubric in `PLAN.md` section by section; name the rubric line.
- After each session, append one log line to `PLAN.md` and tick milestones — do this before
  anything else at the end of a session.
- Scope guard in `PLAN.md` is binding: no Triton, no codegen, no aten lowering, no dynamic shapes.

## Environment

- `uv run python …` / `uv run pytest` — Python 3.12, torch 2.14 CPU-only (GPU deliberately unused here).
