"""Export compact parquet files for the Streamlit app (so the app starts instantly and deploys without raw data)."""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "data"


def export() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(ROOT / "data" / "processed" / "olist.duckdb"), read_only=True)
    con.execute(f"""copy (select order_id, customer_unique_id, customer_state, purchased_at, order_month,
                             order_value, category, review_score, days_vs_estimate
                      from orders) to '{OUT / "orders.parquet"}' (format parquet, compression zstd)""")
    con.execute(f"""copy (select customer_unique_id, customer_state, cohort_month, frequency, monetary,
                             recency_days, segment
                      from customers) to '{OUT / "customers.parquet"}' (format parquet, compression zstd)""")
    for f in sorted(OUT.glob("*.parquet")):
        print(f"{f.name:20s} {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    export()
