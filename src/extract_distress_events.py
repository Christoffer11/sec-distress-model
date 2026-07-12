"""Extract candidate corporate-distress events from SEC filing histories."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "cik",
    "companyName",
    "filingDate",
    "reportDate",
    "form",
    "items",
    "accessionNumber",
    "filingUrl",
]


def extract_item_103(filings: pd.DataFrame) -> pd.DataFrame:
    """Return 8-K filings reporting Item 1.03."""
    required = {"cik", "companyName", "filingDate", "form", "items"}

    missing = required.difference(filings.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    forms = filings["form"].fillna("").str.upper()
    items = filings["items"].fillna("").astype(str)

    is_8k = forms.isin(["8-K", "8-K/A"])

    # The items field may contain several comma-separated item numbers.
    has_item_103 = items.str.contains(
        r"(?<!\d)1\.03(?!\d)",
        regex=True,
    )

    events = filings.loc[is_8k & has_item_103].copy()

    events["filingDate"] = pd.to_datetime(
        events["filingDate"],
        errors="coerce",
    )

    if "reportDate" in events.columns:
        events["reportDate"] = pd.to_datetime(
            events["reportDate"],
            errors="coerce",
        )

    available_columns = [
        column for column in OUTPUT_COLUMNS if column in events.columns
    ]

    return (
        events[available_columns]
        .sort_values(["cik", "filingDate"])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract SEC 8-K Item 1.03 distress disclosures."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="CSV created by fetch_submissions.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/events/distress_events.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    filings = pd.read_csv(
        args.input_file,
        dtype={"cik": str},
    )

    events = extract_item_103(filings)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output, index=False)

    print(f"Candidate Item 1.03 filings found: {len(events):,}")
    print(f"Saved to: {args.output}")

    if not events.empty:
        print()
        print(events.to_string(index=False))


if __name__ == "__main__":
    main()