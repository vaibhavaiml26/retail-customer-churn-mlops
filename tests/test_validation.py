import pandas as pd
from data_validation import validate_schema


def test_invalid_schema_is_rejected():
    df = pd.DataFrame({"Invoice": ["1"]})
    result = validate_schema(df)
    assert not result.passed
    assert "Missing required columns" in result.errors[0]


def test_valid_schema_passes(small_transactions):
    assert validate_schema(small_transactions).passed
