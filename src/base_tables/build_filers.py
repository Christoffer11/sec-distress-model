"""Build the neutral SEC company master table.

The script scans the complete SEC ``submissions.zip`` archive directly and
creates one row per CIK. It applies no modelling filters: companies are kept
regardless of entity type, industry, filing history, or distress status.

Default output
--------------
data/base/companies.parquet

Example
-------
PowerShell:
    python src/base_tables/build_companies.py

Optional CSV output:
    python src/base_tables/build_companies.py `
        --output data/base/companies.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import date
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


CIK_PATTERN = re.compile(r"CIK(\d{1,10})", flags=re.IGNORECASE)
ANNUAL_FORM = "10-K"

DEFAULT_ARCHIVE = Path("data/raw/sec_bulk/submissions.zip")
DEFAULT_OUTPUT = Path("data/base/filers.parquet")


def clean_text(value: object) -> str | None:
    """Convert an SEC value to a clean nullable string."""
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.casefold() in {"nan", "none", "null"}:
        return None

    return text


def normalize_cik(value: object) -> str:
    """Return a CIK as a zero-padded 10-digit string."""
    text = clean_text(value)

    if text is None or not text.isdigit():
        raise ValueError(f"Invalid CIK: {value!r}")

    if len(text) > 10:
        raise ValueError(f"CIK is longer than 10 digits: {text}")

    return text.zfill(10)


def normalize_iso_date(value: object) -> str | None:
    """Return a valid YYYY-MM-DD date string, otherwise None."""
    text = clean_text(value)

    if text is None:
        return None

    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def unique_strings(values: object) -> list[str]:
    """Return unique nonblank strings while preserving their order."""
    if not isinstance(values, list):
        return []

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = clean_text(value)

        if text is not None and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def cik_from_member_name(member_name: str) -> str | None:
    """Extract a CIK from a main or supplementary JSON filename."""
    match = CIK_PATTERN.search(Path(member_name).name)

    if match is None:
        return None

    return match.group(1).zfill(10)


def filing_columns(
    payload: dict[str, Any],
) -> dict[str, list[Any]] | None:
    """Locate filing arrays in a main or historical submissions record."""
    filings = payload.get("filings")

    if isinstance(filings, dict):
        recent = filings.get("recent")

        if isinstance(recent, dict):
            return {
                key: value
                for key, value in recent.items()
                if isinstance(value, list)
            }

    # Supplementary history files contain the arrays at the top level.
    if (
        isinstance(payload.get("accessionNumber"), list)
        and isinstance(payload.get("filingDate"), list)
        and isinstance(payload.get("form"), list)
    ):
        return {
            key: value
            for key, value in payload.items()
            if isinstance(value, list)
        }

    return None


def iter_filings(
    columns: dict[str, list[Any]],
) -> Iterable[tuple[object, object, object]]:
    """Yield accession number, filing date, and form."""
    yield from zip_longest(
        columns.get("accessionNumber", []),
        columns.get("filingDate", []),
        columns.get("form", []),
        fillvalue="",
    )


@dataclass
class CompanySummary:
    """Incrementally accumulated information for one SEC CIK."""

    cik: str

    company_name: str | None = None
    entity_type: str | None = None
    sic: str | None = None
    sic_description: str | None = None
    ein: str | None = None
    lei: str | None = None
    category: str | None = None
    state_of_incorporation: str | None = None
    fiscal_year_end: str | None = None
    tickers: list[str] = field(default_factory=list)
    exchanges: list[str] = field(default_factory=list)

    first_filing_date: str | None = None
    last_filing_date: str | None = None
    first_10k_date: str | None = None
    last_10k_date: str | None = None

    ten_k_accessions: set[str] = field(default_factory=set)
    has_main_record: bool = False
    supplementary_file_count: int = 0

    def update_metadata(self, payload: dict[str, Any]) -> None:
        """Update company-level metadata from a main CIK record."""
        self.has_main_record = True

        self.company_name = (
            clean_text(payload.get("name")) or self.company_name
        )
        self.entity_type = (
            clean_text(payload.get("entityType")) or self.entity_type
        )
        self.sic = clean_text(payload.get("sic")) or self.sic
        self.sic_description = (
            clean_text(payload.get("sicDescription"))
            or self.sic_description
        )
        self.ein = clean_text(payload.get("ein")) or self.ein
        self.lei = clean_text(payload.get("lei")) or self.lei
        self.category = (
            clean_text(payload.get("category")) or self.category
        )
        self.state_of_incorporation = (
            clean_text(payload.get("stateOfIncorporation"))
            or self.state_of_incorporation
        )
        self.fiscal_year_end = (
            clean_text(payload.get("fiscalYearEnd"))
            or self.fiscal_year_end
        )

        payload_tickers = unique_strings(payload.get("tickers"))
        payload_exchanges = unique_strings(payload.get("exchanges"))

        if payload_tickers:
            self.tickers = payload_tickers

        if payload_exchanges:
            self.exchanges = payload_exchanges

    def add_filing(
        self,
        accession_number: object,
        filing_date: object,
        form: object,
    ) -> None:
        """Update filing-history statistics from one filing."""
        accession = clean_text(accession_number)
        filing_date_text = normalize_iso_date(filing_date)
        form_text = (clean_text(form) or "").upper()

        if filing_date_text is not None:
            if (
                self.first_filing_date is None
                or filing_date_text < self.first_filing_date
            ):
                self.first_filing_date = filing_date_text

            if (
                self.last_filing_date is None
                or filing_date_text > self.last_filing_date
            ):
                self.last_filing_date = filing_date_text

        if (
            form_text != ANNUAL_FORM
            or accession is None
            or filing_date_text is None
        ):
            return

        # Count original annual filings once by accession number.
        if accession in self.ten_k_accessions:
            return

        self.ten_k_accessions.add(accession)

        if (
            self.first_10k_date is None
            or filing_date_text < self.first_10k_date
        ):
            self.first_10k_date = filing_date_text

        if (
            self.last_10k_date is None
            or filing_date_text > self.last_10k_date
        ):
            self.last_10k_date = filing_date_text

    def to_row(self) -> dict[str, object]:
        """Return the public company-master representation."""
        return {
            "cik": self.cik,
            "company_name": self.company_name,
            "entity_type": self.entity_type,
            "sic": self.sic,
            "sic_description": self.sic_description,
            "ein": self.ein,
            "lei": self.lei,
            "category": self.category,
            "state_of_incorporation": self.state_of_incorporation,
            "fiscal_year_end": self.fiscal_year_end,
            "tickers": self.tickers,
            "exchanges": self.exchanges,
            "first_filing_date": self.first_filing_date,
            "last_filing_date": self.last_filing_date,
            "first_10k_date": self.first_10k_date,
            "last_10k_date": self.last_10k_date,
            "number_of_10ks": len(self.ten_k_accessions),
            "has_main_record": self.has_main_record,
            "supplementary_file_count": self.supplementary_file_count,
        }


OUTPUT_COLUMNS = [
    "cik",
    "company_name",
    "entity_type",
    "sic",
    "sic_description",
    "ein",
    "lei",
    "category",
    "state_of_incorporation",
    "fiscal_year_end",
    "tickers",
    "exchanges",
    "first_filing_date",
    "last_filing_date",
    "first_10k_date",
    "last_10k_date",
    "number_of_10ks",
    "has_main_record",
    "supplementary_file_count",
]


def scan_submissions_archive(
    archive_path: Path,
    progress_every: int = 5_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Scan all JSON members and return one row per CIK."""
    summaries: dict[str, CompanySummary] = {}

    stats = {
        "json_members": 0,
        "main_records": 0,
        "supplementary_records": 0,
        "invalid_json": 0,
        "members_without_cik": 0,
        "members_without_filings": 0,
    }

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and member.filename.lower().endswith(".json")
        ]

        total = len(members)

        for index, member in enumerate(members, start=1):
            stats["json_members"] += 1

            try:
                with archive.open(member) as source:
                    payload = json.load(source)
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):
                stats["invalid_json"] += 1
                continue

            if not isinstance(payload, dict):
                stats["invalid_json"] += 1
                continue

            cik: str | None = None

            if payload.get("cik") is not None:
                try:
                    cik = normalize_cik(payload["cik"])
                except ValueError:
                    cik = None

            if cik is None:
                cik = cik_from_member_name(member.filename)

            if cik is None:
                stats["members_without_cik"] += 1
                continue

            summary = summaries.setdefault(
                cik,
                CompanySummary(cik=cik),
            )

            is_main_record = isinstance(
                payload.get("filings"),
                dict,
            )

            if is_main_record:
                stats["main_records"] += 1
                summary.update_metadata(payload)
            else:
                stats["supplementary_records"] += 1
                summary.supplementary_file_count += 1

            columns = filing_columns(payload)

            if columns is None:
                stats["members_without_filings"] += 1
            else:
                for filing in iter_filings(columns):
                    summary.add_filing(*filing)

            if (
                progress_every > 0
                and index % progress_every == 0
            ):
                print(
                    f"\rScanned {index:,}/{total:,} JSON members; "
                    f"{len(summaries):,} CIKs",
                    end="",
                    flush=True,
                )

        if progress_every > 0 and total >= progress_every:
            print()

    rows = [
        summary.to_row()
        for summary in summaries.values()
    ]

    companies = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    ).sort_values("cik").reset_index(drop=True)

    for column in (
        "first_filing_date",
        "last_filing_date",
        "first_10k_date",
        "last_10k_date",
    ):
        companies[column] = pd.to_datetime(
            companies[column],
            errors="coerce",
        )

    companies["number_of_10ks"] = (
        companies["number_of_10ks"]
        .astype("int32")
    )
    companies["supplementary_file_count"] = (
        companies["supplementary_file_count"]
        .astype("int16")
    )

    return companies, stats


def write_table(
    companies: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write Parquet or CSV according to the requested suffix."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = output_path.suffix.casefold()

    if suffix == ".parquet":
        try:
            companies.to_parquet(
                output_path,
                index=False,
                engine="pyarrow",
                compression="snappy",
            )
        except ImportError as exc:
            raise RuntimeError(
                "Writing Parquet requires pyarrow. "
                "Install it with: pip install pyarrow"
            ) from exc
    elif suffix == ".csv":
        csv_companies = companies.copy()
        csv_companies["tickers"] = csv_companies[
            "tickers"
        ].map(lambda values: "|".join(values))
        csv_companies["exchanges"] = csv_companies[
            "exchanges"
        ].map(lambda values: "|".join(values))
        csv_companies.to_csv(
            output_path,
            index=False,
        )
    else:
        raise ValueError(
            "--output must end in .parquet or .csv"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one neutral company-master row per SEC CIK."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Input submissions ZIP. Default: {DEFAULT_ARCHIVE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output table. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5_000,
        help=(
            "Print progress after this many JSON members. "
            "Use 0 to disable."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        companies, stats = scan_submissions_archive(
            archive_path=args.archive,
            progress_every=args.progress_every,
        )
        write_table(
            companies=companies,
            output_path=args.output,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Companies / CIKs: {len(companies):,}")
    print(
        "Companies with at least one 10-K: "
        f"{companies['number_of_10ks'].gt(0).sum():,}"
    )
    print(
        "Companies without a 10-K: "
        f"{companies['number_of_10ks'].eq(0).sum():,}"
    )
    print(f"Main records: {stats['main_records']:,}")
    print(
        "Supplementary history records: "
        f"{stats['supplementary_records']:,}"
    )
    print(f"Invalid JSON members skipped: {stats['invalid_json']:,}")
    print(f"Saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
