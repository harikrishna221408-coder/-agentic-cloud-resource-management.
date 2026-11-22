import pytest

from src import app
from tests.helpers import make_project


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """A complete finished run on a tiny synthetic trace (3 machines), shared by the integration and dashboard tests."""
    root = tmp_path_factory.mktemp("proj")
    cfg = make_project(root)
    summary = app.run_all(root=root, cfg=cfg, n_sample=60)
    return root, cfg, summary
