"""Batch-download SEC Company Facts for a list of companies.

Example
-------
PowerShell:
    python src/fetch_many_companyfacts.py data/config/test_ciks.csv `
        --forms 10-K 10-K/A

The input CSV must contain a ``cik`` column. A ``company`` column is optional.
Existing processed files are skipped unless ``--overwrite`` is supplied.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from fetch_companyfacts import (
    fetch_companyfacts_json,
    filter_forms,
    flatten_companyfacts,
)
from fetch_submissions import (
    get_user_agent,
    normalize_cik,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SEC Company Facts for multiple CIKs."
    )
    parser.add_argument(
        "company_file",
        type=Path,
        help="CSV containing at least a column named 'cik'.",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=["10-K", "10-K/A"],
        help="Forms to retain. Default: 10-K and 10-K/A.",
    )
    parser.add_argument(
        "--taxonomy",
        default="us-gaap",
        help=(
            "Taxonomy to retain. Default: us-gaap. "
            "Pass an empty string to retain all taxonomies."
        ),
    )
    parser.add_argument(
        "--user-agent",
        help=(
            "Declared identity for SEC requests. Alternatively set "
            "SEC_USER_AGENT."
        ),
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.25,
        help="Seconds to pause between SEC requests. Default: 0.25.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/companyfacts"),
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/companyfacts"),
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=Path(
            "data/processed/companyfacts/companyfacts_failures.csv"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload companies whose processed file already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds. Default: 60.",
    )
    return parser.parse_args()


def read_companies(path: Path) -> pd.DataFrame:
    """Read and validate the company list."""
    companies = pd.read_csv(path, dtype={"cik": str})

    if "cik" not in companies.columns:
        raise ValueError("Company file must contain a 'cik' column.")

    companies = companies.loc[
        companies["cik"].notna()
    ].copy()

    companies["cik"] = companies["cik"].map(normalize_cik)

    if "company" not in companies.columns:
        companies["company"] = companies["cik"]

    companies = (
        companies.drop_duplicates("cik")
        .reset_index(drop=True)
    )

    if companies.empty:
        raise ValueError("Company file contains no usable CIKs.")

    return companies


def main() -> int:
    args = parse_args()

    if args.pause < 0:
        print("Error: --pause cannot be negative.", file=sys.stderr)
        return 1

    try:
        companies = read_companies(args.company_file)
        user_agent = get_user_agent(args.user_agent)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, str]] = []
    downloaded = 0
    skipped = 0
    total_observations = 0

    taxonomy_filter = args.taxonomy or None

    for index, row in companies.iterrows():
        cik = row["cik"]
        company = str(row["company"])

        raw_path = args.raw_dir / f"CIK{cik}.json"
        processed_path = (
            args.processed_dir / f"CIK{cik}_companyfacts.csv"
        )

        prefix = f"[{index + 1}/{len(companies)}]"

        if processed_path.exists() and not args.overwrite:
            print(f"{prefix} Skipping {company} ({cik}): already exists")
            skipped += 1
            continue

        print(f"{prefix} Fetching {company} ({cik})")

        try:
            payload = fetch_companyfacts_json(
                cik=cik,
                user_agent=user_agent,
                timeout=args.timeout,
            )

            facts = flatten_companyfacts(
                payload,
                taxonomy_filter=taxonomy_filter,
            )
            facts = filter_forms(facts, args.forms)

            save_json(payload, raw_path)
            facts.to_csv(processed_path, index=False)

            downloaded += 1
            total_observations += len(facts)

            print(
                f"  Saved {len(facts):,} observations "
                f"across {facts['concept'].nunique():,} concepts"
            )

        except (
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"  Failed: {exc}")
            failures.append(
                {
                    "cik": cik,
                    "company": company,
                    "error": str(exc),
                }
            )

        if index < len(companies) - 1:
            time.sleep(args.pause)

    if failures:
        args.failure_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pd.DataFrame(failures).to_csv(
            args.failure_output,
            index=False,
        )

    print()
    print(f"Companies requested: {len(companies):,}")
    print(f"Downloaded: {downloaded:,}")
    print(f"Skipped existing: {skipped:,}")
    print(f"Failed: {len(failures):,}")
    print(f"New fact observations: {total_observations:,}")

    if failures:
        print(f"Failure log: {args.failure_output}")

    return 0 if downloaded + skipped > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
