import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest


@pytest.fixture
def small_transactions():
    rows = [
        # Feature-window sales for snapshot 1: [2009-12-01, 2010-05-01)
        ("100", "A", 2, "2009-12-10 10:00", 10.0, 1.0),
        ("101", "A", 1, "2010-01-15 10:00", 15.0, 2.0),
        # Customer 1 has only a return in outcome: should still churn.
        ("C102", "A", -1, "2010-06-01 10:00", 10.0, 1.0),
        # Customer 2 has a positive sale in outcome: should remain active.
        ("103", "A", 1, "2010-06-05 10:00", 20.0, 2.0),
    ]
    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "CustomerID"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df
