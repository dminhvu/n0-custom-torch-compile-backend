"""n0 autograder — run it after every change.

    uv run python -m grader              # full report, scored against PLAN.md's rubric
    uv run python -m grader dce          # only checks matching a keyword (pytest -k syntax)
    uv run python -m grader -v           # also print full tracebacks of failed checks
"""

import argparse
import ast
import contextlib
import io
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import pytest  # noqa: E402

from grader.rubric import AUTO, MANUAL, SECTIONS, grade  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / ".grader" / "history.jsonl"
SHOWN_FAILURES = 4
NOISE = ("TORCHDYNAMO_VERBOSE", "TORCH_LOGS", "from user code:")

_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _color else text


green, red, yellow, dim, bold = (lambda t, k=k: c(t, k) for k in ("32", "31", "33", "2", "1"))


class Collector:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.collect_errors: list[str] = []

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collect_errors.append(str(report.longrepr))

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when == "call" or (report.when == "setup" and not report.passed):
            props = dict(report.user_properties)
            self.checks.append({
                "id": report.nodeid, "rubric": props["rubric"], "title": props["title"],
                "passed": report.passed, "report": report,
            })


def _function_at(path: Path, lineno: int) -> str | None:
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.lineno <= lineno <= (node.end_lineno or node.lineno):
            best = node.name
    return best


def explain(report: pytest.TestReport) -> tuple[str, str | None]:
    """(short reason, 'n0/passes.py:42 in dce' — innermost frame in your code, if any)."""
    lr = report.longrepr
    crash = getattr(lr, "reprcrash", None)
    msg = crash.message if crash else str(lr)
    msg = msg.split("\nassert ")[0].removeprefix("AssertionError: ").strip()
    if "BackendCompilerFailed" in msg:  # Dynamo wraps whatever the backend raised
        msg = msg.split("raised:", 1)[-1].strip()
    msg = "\n".join(l for l in msg.splitlines() if l.strip() and not any(n in l for n in NOISE))
    where = None
    for entry in getattr(getattr(lr, "reprtraceback", None), "reprentries", []):
        loc = getattr(entry, "reprfileloc", None)
        if loc is None:
            continue
        path = (ROOT / loc.path).resolve()
        if path.is_relative_to(ROOT / "n0"):
            fn = _function_at(path, loc.lineno)
            where = f"{path.relative_to(ROOT)}:{loc.lineno}" + (f" in {fn}()" if fn else "")
    if msg.startswith("TimeoutError"):
        where = None  # wherever the alarm happened to land — not where the bug is
    return msg, where


def _todo(msg: str) -> bool:
    return "NotImplementedError" in msg or "not implemented yet" in msg


def load_last() -> dict | None:
    try:
        return json.loads(HISTORY.read_text().splitlines()[-1])
    except (OSError, IndexError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m grader", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keyword", nargs="?", help="only run checks matching this (pytest -k expression)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print full tracebacks of failures")
    args = ap.parse_args()

    os.chdir(ROOT)
    collector = Collector()
    pytest_args = [str(ROOT / "grader"), "-q", "-p", "no:cacheprovider", "--tb=long", "-W", "ignore"]
    if args.keyword:
        pytest_args += ["-k", args.keyword]
    started = time.time()
    print(dim("running checks…"), end="\r", flush=True)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        code = pytest.main(pytest_args, plugins=[collector])
    elapsed = time.time() - started

    import torch

    print(bold(f"n0 grader · torch {torch.__version__} · {time.strftime('%Y-%m-%d %H:%M')} · {elapsed:.0f}s"))
    if collector.collect_errors or (code not in (0, 1) and not collector.checks):
        print(red("\nThe checks could not even be collected — n0/ probably doesn't import:\n"))
        print("\n".join(collector.collect_errors) or sink.getvalue())
        return 2
    if not collector.checks:
        print(yellow(f"\nno checks match {args.keyword!r}"))
        return 1

    by_line: dict[str, list[dict]] = defaultdict(list)
    for chk in collector.checks:
        by_line[chk["rubric"]].append(chk)

    subset = bool(args.keyword)
    auto_total = 0.0
    for sec, (sec_title, sec_points) in SECTIONS.items():
        lines = [(lid, spec) for lid, spec in AUTO.items() if spec[0] == sec and lid in by_line]
        manual = [(lid, spec) for lid, spec in MANUAL.items() if spec[0] == sec]
        if not lines and (subset or not manual):
            continue
        sec_score = 0.0
        body = []
        for lid, (_, title, points) in lines:
            checks = by_line[lid]
            passed = sum(ch["passed"] for ch in checks)
            score = points * passed / len(checks)
            sec_score += score
            mark = green("✓") if passed == len(checks) else (red("✗") if passed == 0 else yellow("◐"))
            body.append(f"  {mark} {lid}  {title:<66} {score:4.1f} / {points:<2}  {dim(f'{passed}/{len(checks)} checks')}")
            failures = [ch for ch in checks if not ch["passed"]]
            todo = [f for f in failures if _todo(explain(f["report"])[0])]
            if todo:
                where = sorted({(explain(f["report"])[1] or "n0/").split(" in ")[-1] for f in todo})
                body.append(dim(f"       · {len(todo)} check(s) waiting on {', '.join(where)} (not implemented yet)"))
            real = [f for f in failures if f not in todo]
            for f in real if args.verbose else real[:SHOWN_FAILURES]:
                msg, where = explain(f["report"])
                body.append(f"       {red('✗')} {f['title']}")
                for i, line in enumerate(msg.splitlines()[:4]):
                    body.append(f"         {'→ ' if i == 0 else '  '}{line}")
                if where:
                    body.append(dim(f"           at {where}"))
                if args.verbose:
                    body.append(dim("\n".join("           " + l for l in f["report"].longreprtext.splitlines())))
            if len(real) > SHOWN_FAILURES and not args.verbose:
                body.append(dim(f"       … {len(real) - SHOWN_FAILURES} more — rerun with -v, or narrow with a keyword"))
        for lid, (_, title, points) in manual:
            body.append(dim(f"  · {lid}  {title:<66}  — / {points}   manual"))
        auto_total += sec_score
        auto_points = sum(p for _, (_, _, p) in lines)
        head = f"{sec} · {sec_title}"
        if subset:
            summary = ""
        elif lines:
            summary = f"{sec_score:.1f} / {auto_points} automated  ({sec_points} total)"
        else:
            summary = f"manual  ({sec_points} total)"
        print(f"\n{bold(head):<40}  {summary}")
        print("\n".join(body))

    if subset:
        print(dim("\n(keyword run — no total; run without a keyword for the score)"))
        return 0 if all(ch["passed"] for ch in collector.checks) else 1

    auto_max = sum(p for _, _, p in AUTO.values())
    manual_max = sum(p for _, _, p in MANUAL.values())
    best = auto_total + manual_max
    print(f"\n{bold('Automated')}   {auto_total:.1f} / {auto_max}")
    print(f"{bold('Best case')}   {best:.1f} / 100 → {grade(best)}   {dim(f'(if the {manual_max} manual points are full marks)')}")

    last = load_last()
    now_passed = sorted(ch["id"] for ch in collector.checks if ch["passed"])
    if last:
        before = set(last["passed"])
        gained = [ch for ch in collector.checks if ch["passed"] and ch["id"] not in before]
        lost = [ch for ch in collector.checks if not ch["passed"] and ch["id"] in before]
        delta = auto_total - last["auto"]
        sign = green(f"+{delta:.1f}") if delta > 0 else (red(f"{delta:.1f}") if delta < 0 else "±0")
        print(f"{bold('Since last')}  {sign}   {len(gained)} newly passing, {len(lost)} newly failing")
        for ch in lost:
            print(red(f"   regression: {ch['rubric']} {ch['title']}"))
    HISTORY.parent.mkdir(exist_ok=True)
    with HISTORY.open("a") as f:
        f.write(json.dumps({"time": time.time(), "auto": round(auto_total, 2), "passed": now_passed}) + "\n")
    return 0 if all(ch["passed"] for ch in collector.checks) else 1


if __name__ == "__main__":
    sys.exit(main())
