"""Download and flatten SEC XBRL company facts for one company.

Examples
--------
PowerShell:
    $env:SEC_USER_AGENT = "Your Name your.email@example.com"
    python src/fetch_companyfacts.py 320193 --forms 10-K 10-K/A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from fetch_submissions import get_user_agent, normalize_cik, save_json

SEC_COMPANYFACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
)
DEFAULT_TIMEOUT_SECONDS = 60


def fetch_companyfacts_json(
    cik: str,
    user_agent: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Fetch all standardized XBRL company facts for one CIK."""
    url = SEC_COMPANYFACTS_URL.format(cik=cik)
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
        status = (
            exc.response.status_code
            if exc.response is not None
            else "unknown"
        )
        raise RuntimeError(
            f"SEC returned HTTP {status} for CIK {cik}. "
            "Check the CIK and declared user agent."
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the SEC: {exc}") from exc

    try:
        return response.json()
    except requests.JSONDecodeError as exc:
        raise RuntimeError("SEC response was not valid JSON.") from exc


def flatten_companyfacts(
    payload: dict[str, Any],
    taxonomy_filter: str | None = "us-gaap",
) -> pd.DataFrame:
    """Convert nested SEC company-facts JSON into a long table."""
    cik = normalize_cik(payload["cik"])
    entity_name = payload.get("entityName")
    facts = payload.get("facts", {})

    rows: list[dict[str, Any]] = []

    for taxonomy, concepts in facts.items():
        if taxonomy_filter and taxonomy != taxonomy_filter:
            continue

        for concept, concept_data in concepts.items():
            label = concept_data.get("label")
            description = concept_data.get("description")
            units = concept_data.get("units", {})

            for unit, observations in units.items():
                for observation in observations:
                    rows.append(
                        {
                            "cik": cik,
                            "companyName": entity_name,
                            "taxonomy": taxonomy,
                            "concept": concept,
                            "label": label,
                            "description": description,
                            "unit": unit,
                            "start": observation.get("start"),
                            "end": observation.get("end"),
                            "value": observation.get("val"),
                            "accessionNumber": observation.get("accn"),
                            "fiscalYear": observation.get("fy"),
                            "fiscalPeriod": observation.get("fp"),
                            "form": observation.get("form"),
                            "filed": observation.get("filed"),
                            "frame": observation.get("frame"),
                        }
                    )

    columns = [
        "cik",
        "companyName",
        "taxonomy",
        "concept",
        "label",
        "description",
        "unit",
        "start",
        "end",
        "value",
        "accessionNumber",
        "fiscalYear",
        "fiscalPeriod",
        "form",
        "filed",
        "frame",
    ]

    facts_long = pd.DataFrame(rows, columns=columns)

    for column in ("start", "end", "filed"):
        facts_long[column] = pd.to_datetime(
            facts_long[column], errors="coerce"
        )

    facts_long["fiscalYear"] = pd.to_numeric(
        facts_long["fiscalYear"], errors="coerce"
    ).astype("Int64")

    return facts_long


def filter_forms(
    facts: pd.DataFrame,
    forms: list[str] | None,
) -> pd.DataFrame:
    """Optionally retain facts reported on selected SEC forms."""
    if not forms:
        return facts

    wanted = {form.upper() for form in forms}
    return facts.loc[
        facts["form"].fillna("").str.upper().isin(wanted)
    ].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SEC XBRL company facts for one CIK."
    )
    parser.add_argument(
        "cik",
        help="Company CIK, with or without leading zeros.",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        help="Optional form filter, e.g. --forms 10-K 10-K/A.",
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
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cik = normalize_cik(args.cik)
        user_agent = get_user_agent(args.user_agent)

        payload = fetch_companyfacts_json(
            cik=cik,
            user_agent=user_agent,
            timeout=args.timeout,
        )

        taxonomy_filter = args.taxonomy or None
        facts = flatten_companyfacts(
            payload,
            taxonomy_filter=taxonomy_filter,
        )
        facts = filter_forms(facts, args.forms)

        raw_path = args.raw_dir / f"CIK{cik}.json"
        csv_path = args.processed_dir / f"CIK{cik}_companyfacts.csv"

        save_json(payload, raw_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        facts.to_csv(csv_path, index=False)

    except (ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Company: {payload.get('entityName', 'Unknown company')}")
    print(f"CIK: {cik}")
    print(f"Fact observations saved: {len(facts):,}")
    print(f"Distinct concepts: {facts['concept'].nunique():,}")
    print(f"Raw JSON: {raw_path}")
    print(f"Processed CSV: {csv_path}")

    if not facts.empty:
        preview_columns = [
            "end",
            "filed",
            "form",
            "fiscalYear",
            "fiscalPeriod",
            "concept",
            "unit",
            "value",
        ]
        print("\nMost recently filed observations:")
        print(
            facts.sort_values("filed", ascending=False)[preview_columns]
            .head(10)
            .to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
