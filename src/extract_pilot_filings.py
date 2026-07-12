"""Extract pilot filing histories from the SEC bulk submissions ZIP.

The Company Facts archive contains financial observations, but the modelling
dataset is anchored on filing histories. This script extracts the pilot CIKs'
10-K, 10-K/A, 8-K, and 8-K/A filings from ``submissions.zip`` and creates the
matching Item 1.03 distress-event table.

Example
-------
PowerShell:
    python src/extract_pilot_filings.py

Outputs
-------
- data/processed/filings/pilot_filings.csv
- data/processed/events/pilot_distress_events.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CIK_PATTERN = re.compile(r"CIK(\d{10})", flags=re.IGNORECASE)
ITEM_103_PATTERN = re.compile(r"(?<![\d.])1\.03(?![\d.])")

DEFAULT_PILOT = Path(
    "data/processed/universe/pilot_universe.csv"
)
DEFAULT_ARCHIVE = Path(
    "data/raw/sec_bulk/submissions.zip"
)
DEFAULT_FILINGS_OUTPUT = Path(
    "data/processed/filings/pilot_filings.csv"
)
DEFAULT_EVENTS_OUTPUT = Path(
    "data/processed/events/pilot_distress_events.csv"
)
DEFAULT_MISSING_OUTPUT = Path(
    "data/processed/filings/pilot_filings_missing.csv"
)
DEFAULT_FAILURE_OUTPUT = Path(
    "data/processed/filings/pilot_filings_failures.csv"
)

DEFAULT_FORMS = ["10-K", "10-K/A", "8-K", "8-K/A"]

FILING_FIELDS = [
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "act",
    "form",
    "fileNumber",
    "filmNumber",
    "items",
    "size",
    "isXBRL",
    "isInlineXBRL",
    "primaryDocument",
    "primaryDocDescription",
]

EVENT_OUTPUT_COLUMNS = [
    "cik",
    "companyName",
    "filingDate",
    "reportDate",
    "form",
    "items",
    "accessionNumber",
    "primaryDocument",
]


def normalize_cik(cik: str | int) -> str:
    """Return a CIK as a zero-padded 10-digit string."""
    cik_text = str(cik).strip()

    if not cik_text.isdigit():
        raise ValueError("CIK must contain digits only.")

    if len(cik_text) > 10:
        raise ValueError("CIK cannot be longer than 10 digits.")

    return cik_text.zfill(10)


def read_pilot(path: Path) -> pd.DataFrame:
    """Read pilot CIKs and company names."""
    pilot = pd.read_csv(path, dtype={"cik": str})

    if "cik" not in pilot.columns:
        raise ValueError(
            "Pilot universe must contain a 'cik' column."
        )

    pilot = pilot.loc[pilot["cik"].notna()].copy()
    pilot["cik"] = pilot["cik"].map(normalize_cik)

    if "company" not in pilot.columns:
        pilot["company"] = ""

    return (
        pilot[["cik", "company"]]
        .drop_duplicates("cik")
        .reset_index(drop=True)
    )


def cik_from_member_name(member_name: str) -> str | None:
    """Extract a 10-digit CIK from a ZIP member filename."""
    match = CIK_PATTERN.search(Path(member_name).name)
    return match.group(1) if match else None


def filing_columns(
    payload: dict[str, Any],
) -> dict[str, list[Any]] | None:
    """Locate filing arrays in a main or historical SEC JSON file."""
    filings = payload.get("filings")

    if isinstance(filings, dict):
        recent = filings.get("recent")
        if isinstance(recent, dict):
            return {
                key: value
                for key, value in recent.items()
                if isinstance(value, list)
            }

    if (
        isinstance(payload.get("accessionNumber"), list)
        and isinstance(payload.get("form"), list)
    ):
        return {
            key: value
            for key, value in payload.items()
            if isinstance(value, list)
        }

    return None


def iter_filing_rows(
    columns: dict[str, list[Any]],
    wanted_forms: set[str],
) -> Iterable[dict[str, Any]]:
    """Yield selected filings from column-oriented SEC arrays."""
    arrays = [
        columns.get(field, [])
        for field in FILING_FIELDS
    ]

    for values in zip_longest(*arrays, fillvalue=""):
        row = dict(zip(FILING_FIELDS, values))
        form = str(row.get("form", "")).strip().upper()

        if form in wanted_forms:
            yield row


def extract_item_103(
    filings: pd.DataFrame,
) -> pd.DataFrame:
    """Return 8-K and 8-K/A filings reporting Item 1.03."""
    required = {
        "cik",
        "companyName",
        "filingDate",
        "form",
        "items",
    }
    missing = required.difference(filings.columns)

    if missing:
        raise ValueError(
            f"Filings are missing columns: {sorted(missing)}"
        )

    forms = filings["form"].fillna("").str.upper()
    items = filings["items"].fillna("").astype(str)

    events = filings.loc[
        forms.isin(["8-K", "8-K/A"])
        & items.str.contains(ITEM_103_PATTERN, regex=True)
    ].copy()

    available_columns = [
        column
        for column in EVENT_OUTPUT_COLUMNS
        if column in events.columns
    ]

    return (
        events[available_columns]
        .sort_values(["cik", "filingDate", "accessionNumber"])
        .reset_index(drop=True)
    )


def extract_pilot_filings(
    pilot: pd.DataFrame,
    archive_path: Path,
    forms: list[str],
    progress_every: int = 10_000,
) -> tuple[pd.DataFrame, set[str], list[dict[str, str]]]:
    """Scan submissions.zip and retain selected filings for pilot CIKs."""
    wanted_ciks = set(pilot["cik"])
    wanted_forms = {form.upper() for form in forms}
    company_names = dict(zip(pilot["cik"], pilot["company"]))

    rows: list[dict[str, Any]] = []
    found_ciks: set[str] = set()
    failures: list[dict[str, str]] = []

    with zipfile.ZipFile(archive_path) as archive:
        json_members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and member.filename.lower().endswith(".json")
        ]

        total_members = len(json_members)

        for index, member in enumerate(json_members, start=1):
            cik = cik_from_member_name(member.filename)

            if cik not in wanted_ciks:
                if (
                    progress_every > 0
                    and index % progress_every == 0
                ):
                    print(
                        f"Scanned {index:,}/{total_members:,} "
                        "archive members"
                    )
                continue

            try:
                with archive.open(member) as source:
                    payload = json.load(source)
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
            ) as exc:
                failures.append(
                    {
                        "cik": cik or "",
                        "member": member.filename,
                        "error": str(exc),
                    }
                )
                continue

            if not isinstance(payload, dict):
                failures.append(
                    {
                        "cik": cik or "",
                        "member": member.filename,
                        "error": "JSON payload is not an object.",
                    }
                )
                continue

            found_ciks.add(cik)

            if "filings" in payload:
                payload_name = str(
                    payload.get("name", "")
                ).strip()

                if payload_name:
                    company_names[cik] = payload_name

            columns = filing_columns(payload)

            if columns is not None:
                tickers = payload.get("tickers", [])
                exchanges = payload.get("exchanges", [])

                ticker_text = (
                    ",".join(str(value) for value in tickers)
                    if isinstance(tickers, list)
                    else ""
                )
                exchange_text = (
                    ",".join(str(value) for value in exchanges)
                    if isinstance(exchanges, list)
                    else ""
                )

                for filing in iter_filing_rows(
                    columns=columns,
                    wanted_forms=wanted_forms,
                ):
                    rows.append(
                        {
                            "cik": cik,
                            "companyName": company_names.get(
                                cik,
                                "",
                            ),
                            "tickers": ticker_text,
                            "exchanges": exchange_text,
                            **filing,
                        }
                    )

            if (
                progress_every > 0
                and index % progress_every == 0
            ):
                print(
                    f"Scanned {index:,}/{total_members:,} "
                    "archive members"
                )

    output_columns = [
        "cik",
        "companyName",
        "tickers",
        "exchanges",
        *FILING_FIELDS,
    ]
    filings = pd.DataFrame(
        rows,
        columns=output_columns,
    )

    if filings.empty:
        return filings, found_ciks, failures

    filings["cik"] = filings["cik"].map(normalize_cik)

    for column in (
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
    ):
        filings[column] = pd.to_datetime(
            filings[column],
            errors="coerce",
        )

    filings = (
        filings.dropna(
            subset=[
                "cik",
                "accessionNumber",
                "filingDate",
                "form",
            ]
        )
        .drop_duplicates(
            subset=["cik", "accessionNumber"],
            keep="first",
        )
        .sort_values(
            ["cik", "filingDate", "accessionNumber"]
        )
        .reset_index(drop=True)
    )

    # Historical supplementary files do not include company metadata.
    filings["companyName"] = filings["cik"].map(
        company_names
    )

    return filings, found_ciks, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pilot 10-K and 8-K histories from the SEC "
            "bulk submissions archive."
        )
    )
    parser.add_argument(
        "--pilot",
        type=Path,
        default=DEFAULT_PILOT,
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
    )
    parser.add_argument(
        "--filings-output",
        type=Path,
        default=DEFAULT_FILINGS_OUTPUT,
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        default=DEFAULT_EVENTS_OUTPUT,
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=DEFAULT_FORMS,
        help=(
            "Forms to retain. Default: "
            "10-K 10-K/A 8-K 8-K/A."
        ),
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=DEFAULT_MISSING_OUTPUT,
    )
    parser.add_argument(
        "--failure-output",
        type=Path,
        default=DEFAULT_FAILURE_OUTPUT,
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        pilot = read_pilot(args.pilot)

        filings, found_ciks, failures = extract_pilot_filings(
            pilot=pilot,
            archive_path=args.archive,
            forms=args.forms,
            progress_every=args.progress_every,
        )

        events = extract_item_103(filings)

        args.filings_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.events_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        filings.to_csv(
            args.filings_output,
            index=False,
        )
        events.to_csv(
            args.events_output,
            index=False,
        )

        missing_ciks = sorted(
            set(pilot["cik"]) - found_ciks
        )

        if missing_ciks:
            missing = pilot.loc[
                pilot["cik"].isin(missing_ciks)
            ].copy()
            missing["reason"] = (
                "CIK not found in submissions.zip"
            )
            args.missing_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            missing.to_csv(
                args.missing_output,
                index=False,
            )

        if failures:
            args.failure_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            pd.DataFrame(failures).to_csv(
                args.failure_output,
                index=False,
            )

    except (
        FileNotFoundError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Pilot companies: {len(pilot):,}")
    print(f"Companies found in archive: {len(found_ciks):,}")
    print(f"Selected filings: {len(filings):,}")
    print(
        "Companies with at least one selected filing: "
        f"{filings['cik'].nunique():,}"
    )
    print(f"Item 1.03 filings: {len(events):,}")
    print(f"Missing companies: {len(missing_ciks):,}")
    print(f"Failed archive members: {len(failures):,}")
    print(f"Filings saved to: {args.filings_output}")
    print(f"Events saved to: {args.events_output}")

    return 0 if not filings.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
