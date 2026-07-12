"""Fetch SEC submission histories for several companies."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from extract_distress_events import extract_item_103
from fetch_submissions import (
    fetch_submission_json,
    get_user_agent,
    normalize_cik,
    save_json,
    submissions_to_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch SEC submissions for a list of CIKs."
    )
    parser.add_argument(
        "company_file",
        type=Path,
        help="CSV containing a column named 'cik'.",
    )
    parser.add_argument(
        "--user-agent",
        help="SEC user agent. Alternatively set SEC_USER_AGENT.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.25,
        help="Seconds to pause between SEC requests.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/submissions"),
    )
    parser.add_argument(
        "--filings-output",
        type=Path,
        default=Path("data/processed/filings/all_filings.csv"),
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        default=Path("data/processed/events/distress_events.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    user_agent = get_user_agent(args.user_agent)

    companies = pd.read_csv(
        args.company_file,
        dtype={"cik": str},
    )

    if "cik" not in companies.columns:
        raise ValueError("Company file must contain a 'cik' column.")

    filing_tables: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []

    for index, row in companies.iterrows():
        cik = normalize_cik(row["cik"])
        company_label = row.get("company", cik)

        print(
            f"[{index + 1}/{len(companies)}] "
            f"Fetching {company_label} ({cik})"
        )

        try:
            payload = fetch_submission_json(
                cik=cik,
                user_agent=user_agent,
            )

            save_json(
                payload,
                args.raw_dir / f"CIK{cik}.json",
            )

            filings = submissions_to_dataframe(payload)
            filing_tables.append(filings)

            print(f"  Retrieved {len(filings):,} filings")

        except (ValueError, RuntimeError) as exc:
            print(f"  Failed: {exc}")
            failures.append(
                {
                    "cik": cik,
                    "company": str(company_label),
                    "error": str(exc),
                }
            )

        if index < len(companies) - 1:
            time.sleep(args.pause)

    if not filing_tables:
        print("No filing histories were successfully retrieved.")
        return 1

    all_filings = pd.concat(
        filing_tables,
        ignore_index=True,
    )

    all_filings = (
        all_filings
        .drop_duplicates(
            subset=["cik", "accessionNumber"],
        )
        .sort_values(["cik", "filingDate"])
        .reset_index(drop=True)
    )

    distress_events = extract_item_103(all_filings)

    args.filings_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.events_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_filings.to_csv(
        args.filings_output,
        index=False,
    )
    distress_events.to_csv(
        args.events_output,
        index=False,
    )

    if failures:
        failure_path = Path(
            "data/processed/filings/fetch_failures.csv"
        )
        failure_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pd.DataFrame(failures).to_csv(
            failure_path,
            index=False,
        )
        print(f"Failures saved: {failure_path}")

    print()
    print(f"Companies requested: {len(companies):,}")
    print(f"Companies retrieved: {len(filing_tables):,}")
    print(f"Total filings: {len(all_filings):,}")
    print(f"Distress disclosures: {len(distress_events):,}")
    print(f"All filings: {args.filings_output}")
    print(f"Events: {args.events_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())