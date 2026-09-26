"""PLAN.md's rubric, split into what the grader can check and what needs a human."""

SECTIONS = {
    "A": ("Correctness", 30),
    "B": ("The passes", 30),
    "C": ("Engineering", 15),
    "D": ("Understanding", 25),
}

# line id -> (section, title, points). Every automated check is tagged with one of these ids.
AUTO = {
    "A1": ("A", "compiled output allclose to eager on every test module", 10),
    "A2": ("A", "passes are idempotent (second run changes nothing)", 10),
    "A3": ("A", "in-place ops survive DCE, result still correct", 5),
    "A4": ("A", "reference eliminate_dead_code() finds nothing after your DCE", 5),
    "B1": ("B", "DCE: reverse walk, side-effect check, no special-casing by name", 10),
    "B2": ("B", "folding: recursive, no nondeterminism, size cap, get_attr buffers", 10),
    "B3": ("B", "CSE: value numbering, kwargs + list args, only pure nodes merged", 10),
    "C1": ("C", "core ≤ ~300 lines, one function per pass, type hints", 5),
    "C2": ("C", "pytest green, one test file per pass, one torch.compile test", 5),
}

MANUAL = {
    "C3": ("C", "README (numbers, freezing caveat) + commit history — ask Claude", 5),
    "D": ("D", "10 questions aloud, no notes — ask Claude to quiz you", 25),
}


def grade(score: float) -> str:
    for threshold, g in ((90, "1.0"), (80, "1.7"), (70, "2.3"), (60, "3.0")):
        if score >= threshold:
            return g
    return "redo the weak section"
