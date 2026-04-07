"""One-off: create a starter PQL_to_SQL_Function_Reference.xlsx if missing. Replace with your full mapping."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
out = ROOT / "data" / "raw" / "PQL_to_SQL_Function_Reference.xlsx"
out.parent.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(
    [
        ["TO_FLOAT(x)", "CAST(x AS DOUBLE)"],
        ['COUNT_TABLE("T")', "(SELECT COUNT(*) FROM T)"],
        ["SUM(...)", "SUM(...)"],
    ],
    columns=["PQL_construct", "SQL_Spark_equivalent"],
)
df.to_excel(out, index=False)
print(f"Wrote {out}")
