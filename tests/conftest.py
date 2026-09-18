"""Working precision is a global setting in mpmath, so a module level
assignment in one test file is overwritten by the import of the next.  Each
test therefore sets the precision it needs, through this fixture, rather than
inheriting whatever the last import happened to leave behind.
"""
import mpmath as mp
import pytest

DPS = {
    "test_parametric": 30,
    "test_universality": 25,
    "test_meanfield": 20,
    "test_cumulant": 15,
}


@pytest.fixture(autouse=True)
def working_precision(request):
    name = request.module.__name__.rsplit(".", 1)[-1]
    old = mp.mp.dps
    mp.mp.dps = DPS.get(name, 25)
    yield
    mp.mp.dps = old
