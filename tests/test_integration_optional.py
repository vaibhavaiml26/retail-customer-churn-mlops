import importlib.util
import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("pyarrow") is None,
    reason="pyarrow is required for the Parquet-backed integration pipeline",
)


def test_parquet_dependency_available_for_full_pipeline():
    import pyarrow  # noqa: F401
