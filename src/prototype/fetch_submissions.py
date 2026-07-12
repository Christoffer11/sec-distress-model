"""Download a company's recent SEC filing history.

Example
-------
PowerShell:
    $env:SEC_USER_AGENT = "Your Name your.email@example.com"
    python src/fetch_submissions.py 320193 --forms 10-K 10-Q 8-K

The SEC submissions endpoint returns at least one year or 1,000 of the most
recent filings, whichever is greater. This first version intentionally fetches
only the filings contained in ``filings.recent``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_TIMEOUT_SECONDS = 30


def normalize_cik(cik: str | int) -> str:
    """Return a CIK as a zero-padded 10-digit string."""
    cik_text = str(cik).strip()

    if not cik_text.isdigit():
        raise ValueError("CIK must contain digits only.")

    if len(cik_text) > 10:
        raise ValueError("CIK cannot be longer than 10 digits.")

    return cik_text.zfill(10)


def get_user_agent(cli_user_agent: str | None) -> str:
    """Get the SEC user agent from the CLI argument or environment."""
    user_agent = cli_user_agent or os.getenv("SEC_USER_AGENT")

    if not user_agent:
        raise ValueError(
            "A declared SEC user agent is required. Pass --user-agent "
            "'Your Name your.email@example.com' or set SEC_USER_AGENT."
        )

    return user_agent.strip()


def fetch_submission_json(
    cik: str,
    user_agent: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Fetch the SEC submissions JSON for one company."""
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }

    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.Timeout as exc:
        raise RuntimeError(
            f"SEC request timed out after {timeout} seconds."
        ) from exc
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise RuntimeError(
            f"SEC returned HTTP {status} for CIK {cik}. "
            "Check the CIK and your declared user agent."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the SEC: {exc}") from exc

    try:
        return response.json()
    except requests.JSONDecodeError as exc:
        raise RuntimeError("SEC response was not valid JSON.") from exc


def submissions_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert ``filings.recent`` from column-oriented JSON to a DataFrame."""
    recent = payload.get("filings", {}).get("recent")

    if not isinstance(recent, dict) or not recent:
        raise ValueError("The SEC response contains no recent filing data.")

    filings = pd.DataFrame(recent)

    if filings.empty:
        return filings

    cik = normalize_cik(payload["cik"])
    company_name = payload.get("name")
    tickers = payload.get("tickers") or []
    exchanges = payload.get("exchanges") or []

    filings.insert(0, "cik", cik)
    filings.insert(1, "companyName", company_name)
    filings.insert(2, "tickers", ",".join(tickers))
    filings.insert(3, "exchanges", ",".join(exchanges))

    accession_without_dashes = filings["accessionNumber"].str.replace(
        "-", "", regex=False
    )
    cik_without_leading_zeros = str(int(cik))

    filings["filingUrl"] = (
        SEC_ARCHIVES_URL
        + "/"
        + cik_without_leading_zeros
        + "/"
        + accession_without_dashes
        + "/"
        + filings["primaryDocument"]
    )

    for column in ("filingDate", "reportDate", "acceptanceDateTime"):
        if column in filings.columns:
            filings[column] = pd.to_datetime(
                filings[column], errors="coerce"
            )

    return filings


def filter_forms(
    filings: pd.DataFrame,
    forms: list[str] | None,
) -> pd.DataFrame:
    """Optionally retain selected SEC form types."""
    if not forms:
        return filings

    wanted_forms = {form.upper() for form in forms}
    return filings.loc[
        filings["form"].str.upper().isin(wanted_forms)
    ].copy()


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save raw JSON, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def save_csv(filings: pd.DataFrame, path: Path) -> None:
    """Save the filing table as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    filings.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download recent SEC submissions for one CIK."
    )
    parser.add_argument(
        "cik",
        help="Company CIK, with or without leading zeros.",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        help="Optional form filter, for example: --forms 10-K 10-Q 8-K",
    )
    parser.add_argument(
        "--user-agent",
        help=(
            "Declared identity for SEC requests. Alternatively set the "
            "SEC_USER_AGENT environment variable."
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/submissions"),
        help="Directory for raw SEC JSON.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed/filings"),
        help="Directory for processed filing CSV files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cik = normalize_cik(args.cik)
        user_agent = get_user_agent(args.user_agent)
        payload = fetch_submission_json(
            cik=cik,
            user_agent=user_agent,
            timeout=args.timeout,
        )
        filings = submissions_to_dataframe(payload)
        filings = filter_forms(filings, args.forms)

        raw_path = args.raw_dir / f"CIK{cik}.json"
        csv_path = args.processed_dir / f"CIK{cik}_filings.csv"

        save_json(payload, raw_path)
        save_csv(filings, csv_path)

    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    company_name = payload.get("name", "Unknown company")
    print(f"Company: {company_name}")
    print(f"CIK: {cik}")
    print(f"Filings saved: {len(filings):,}")
    print(f"Raw JSON: {raw_path}")
    print(f"Processed CSV: {csv_path}")

    if not filings.empty:
        display_columns = [
            column
            for column in (
                "filingDate",
                "reportDate",
                "form",
                "accessionNumber",
                "primaryDocument",
            )
            if column in filings.columns
        ]
        print("\nMost recent filings:")
        print(filings[display_columns].head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
