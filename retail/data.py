"""Validate the supplied file and make the uncertain sales grain explicit."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset.csv"
REQUIRED = {
    "C520_DATE", "C520_STORENR", "C520_ITEMNR", "C520_SOLDPIECES",
    "C520_RETAILPRICE", "ARTNRVIER", "BEZ_LANG", "DISCOUNT",
}


def load_data(path=DATA_PATH):
    return validate_data(pd.read_csv(path))


def validate_data(frame):
    df = frame.copy()
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("The dataset is empty.")
    df["C520_DATE"] = pd.to_datetime(df["C520_DATE"], errors="raise")
    if df[list(REQUIRED)].isna().any().any():
        raise ValueError("Required fields contain missing values.")
    if not df["C520_DATE"].eq(df["C520_DATE"].dt.normalize()).all():
        raise ValueError("Expected daily dates without time components.")
    for col in ["C520_SOLDPIECES", "C520_RETAILPRICE"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(df[col]).all() or (df[col] < 0).any():
            raise ValueError(f"{col} must contain finite, non-negative values.")
    if df["C520_STORENR"].nunique() != 1:
        raise ValueError("This analysis expects one store; filter a store explicitly first.")
    if (df.groupby("ARTNRVIER")["BEZ_LANG"].nunique() > 1).any():
        raise ValueError("A parent product maps to multiple names; resolve the mapping first.")
    if not df["DISCOUNT"].isin([0, 1]).all():
        raise ValueError("DISCOUNT must be a binary flag.")
    return df


def sales_signature(df):
    # Preserve distinct prices, quantities, statuses and change records. Do not
    # deduplicate merely on date/item: some dates contain two different records.
    return [c for c in df if c.startswith("C520_")]


def audit_data(df):
    signature = sales_signature(df)
    distinct = df.drop_duplicates(signature)
    counts = df.groupby(signature, dropna=False)["DISCOUNT"].nunique()
    raw_totals = df.groupby("ARTNRVIER")["C520_SOLDPIECES"].sum()
    unique_totals = distinct.groupby("ARTNRVIER")["C520_SOLDPIECES"].sum()
    products = pd.DataFrame({"raw_sales": raw_totals, "distinct_record_sales": unique_totals})
    products["row_sum_multiplier"] = products.raw_sales.div(
        products.distinct_record_sales.replace(0, np.nan)
    )
    products["product"] = df.groupby("ARTNRVIER")["BEZ_LANG"].first()
    products.index.name = "product_id"
    return {
        "raw_rows": len(df), "distinct_sales_records": len(distinct),
        "repeated_sales_records": len(df) - len(distinct),
        "discount_conflict_records": int((counts > 1).sum()),
        "products": df.ARTNRVIER.nunique(), "stores": df.C520_STORENR.nunique(),
        "start": str(df.C520_DATE.min().date()), "end": str(df.C520_DATE.max().date()),
    }, products.reset_index()


def daily_sales(df, mode="distinct"):
    """Return daily observations; neither interpretation is source-verified.

    distinct: count each identical C520 sales record once. Metadata flags that
    disagree are combined using max for descriptive analysis only.
    raw: retain the original notebook's row-summing assumption for sensitivity.
    """
    if mode == "distinct":
        signature = sales_signature(df)
        records = df.drop_duplicates(signature).drop(columns="DISCOUNT")
        flags = df.groupby(signature, dropna=False, as_index=False)["DISCOUNT"].max()
        records = records.merge(flags, on=signature, how="left", validate="one_to_one")
    elif mode == "raw":
        records = df.copy()
    else:
        raise ValueError("Sales mode must be 'distinct' or 'raw'.")
    out = records.groupby(["C520_DATE", "ARTNRVIER", "BEZ_LANG"], as_index=False).agg(
        sales=("C520_SOLDPIECES", "sum"),
        price=("C520_RETAILPRICE", "mean"),
        any_discount_flag=("DISCOUNT", "max"),
    )
    return out.rename(columns={"C520_DATE": "date", "ARTNRVIER": "product_id", "BEZ_LANG": "product"})


def calendar_series(daily, product_id, start, end):
    part = daily[daily.product_id == product_id].set_index("date")
    # Missing observations remain unknown, including before/after assortment.
    out = part[["sales", "price", "any_discount_flag"]].reindex(pd.date_range(start, end))
    out.index.name = "date"
    return out
