import signal
import warnings

import pytest

warnings.filterwarnings("ignore")

import torch  # noqa: E402

from grader.rubric import AUTO  # noqa: E402

TIME_LIMIT_S = 30
AFTER_TIMEOUT_S = 5
_timeouts: list[str] = []


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "rubric(line): the rubric line (grader/rubric.py) a check counts toward")
    config.addinivalue_line("markers", "time_limit(seconds): override the per-check time limit")
    config.addinivalue_line("markers", "fixpoint: runs optimize(); fails fast once optimize() has hung twice")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        marker = item.get_closest_marker("rubric")
        assert marker and marker.args[0] in AUTO, f"{item.nodeid}: missing/unknown rubric marker"
        doc = (item.function.__doc__ or item.name).strip().splitlines()[0]
        param = f" [{item.callspec.id}]" if hasattr(item, "callspec") else ""
        item.user_properties += [("rubric", marker.args[0]), ("title", doc + param)]


@pytest.fixture(autouse=True)
def _guard(request: pytest.FixtureRequest):
    if len(_timeouts) >= 2 and request.node.get_closest_marker("fixpoint"):
        pytest.fail("not run: optimize() already timed out twice — fix the non-termination first")
    marker = request.node.get_closest_marker("time_limit")
    # after one check has timed out, the rest fail fast instead of each waiting the full limit
    limit = marker.args[0] if marker else (AFTER_TIMEOUT_S if _timeouts else TIME_LIMIT_S)

    def timeout(*_):
        _timeouts.append(request.node.nodeid)
        raise TimeoutError(
            f"check took over {limit}s — a pass or the fixpoint loop does not terminate "
            "(does every pass return False when it changed nothing?)"
        )

    old = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(limit)
    torch._dynamo.reset()
    torch.manual_seed(0)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, old)
