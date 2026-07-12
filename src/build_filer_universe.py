"""Build a historical SEC filer universe from submissions.zip.

The script scans JSON members directly inside the SEC bulk submissions archive.
It summarizes annual 10-K coverage and candidate bankruptcy/receivership
disclosures reported under Form 8-K Item 1.03.

Examples
--------
PowerShell:
    python src/build_filer_universe.py

    python src/build_filer_universe.py `
        --min-10ks 3 `
        --analysis-start 2009-01-01

Outputs
-------
- data/processed/universe/filer_universe.csv
- data/processed/universe/model_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import date
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable


CIK_PATTERN = re.compile(r"CIK(\d{10})", flags=re.IGNORECASE)
ITEM_103_PATTERN = re.compile(
    r"(?<![\d.])1\.03(?![\d.])"
)
ANNUAL_FORM = "10-K"
DISTRESS_FORMS = {"8-K", "8-K/A"}


@dataclass
class FilerSummary:
    """Incremental summary for one SEC filer."""

    cik: str
    company: str | None = None
    entity_type: str | None = None
    sic: str | None = None
    sic_description: str | None = None
    fiscal_year_end: str | None = None
    state_of_incorporation: str | None = None
    tickers: list[str] = field(default_factory=list)
    exchanges: list[str] = field(default_factory=list)

    first_filing_date: str | None = None
    last_filing_date: str | None = None

    ten_k_dates_by_accession: dict[str, str] = field(
        default_factory=dict
    )
    item_103_dates_by_accession: dict[str, str] = field(
        default_factory=dict
    )

    def update_metadata(self, payload: dict[str, Any]) -> None:
        """Update entity metadata from a main CIK JSON file."""
        self.company = clean_text(payload.get("name")) or self.company
        self.entity_type = (
            clean_text(payload.get("entityType")) or self.entity_type
        )
        self.sic = clean_text(payload.get("sic")) or self.sic
        self.sic_description = (
            clean_text(payload.get("sicDescription"))
            or self.sic_description
        )
        self.fiscal_year_end = (
            clean_text(payload.get("fiscalYearEnd"))
            or self.fiscal_year_end
        )
        self.state_of_incorporation = (
            clean_text(payload.get("stateOfIncorporation"))
            or self.state_of_incorporation
        )

        tickers = payload.get("tickers")
        if isinstance(tickers, list):
            self.tickers = unique_strings(tickers)

        exchanges = payload.get("exchanges")
        if isinstance(exchanges, list):
            self.exchanges = unique_strings(exchanges)

    def add_filing(
        self,
        accession: object,
        filing_date: object,
        form: object,
        items: object,
    ) -> None:
        """Update this filer with one filing observation."""
        accession_text = clean_text(accession)
        date_text = normalize_iso_date(filing_date)
        form_text = clean_text(form).upper()
        items_text = clean_text(items)

        if date_text:
            if (
                self.first_filing_date is None
                or date_text < self.first_filing_date
            ):
                self.first_filing_date = date_text

            if (
                self.last_filing_date is None
                or date_text > self.last_filing_date
            ):
                self.last_filing_date = date_text

        if not accession_text or not date_text:
            return

        if form_text == ANNUAL_FORM:
            self.ten_k_dates_by_accession.setdefault(
                accession_text,
                date_text,
            )

        if (
            form_text in DISTRESS_FORMS
            and ITEM_103_PATTERN.search(items_text)
        ):
            self.item_103_dates_by_accession.setdefault(
                accession_text,
                date_text,
            )

    def to_row(self) -> dict[str, object]:
        """Flatten the summary to one CSV row."""
        ten_k_dates = sorted(self.ten_k_dates_by_accession.values())
        item_103_dates = sorted(
            self.item_103_dates_by_accession.values()
        )

        return {
            "cik": self.cik,
            "company": self.company or "",
            "entityType": self.entity_type or "",
            "sic": self.sic or "",
            "sicDescription": self.sic_description or "",
            "fiscalYearEnd": self.fiscal_year_end or "",
            "stateOfIncorporation": (
                self.state_of_incorporation or ""
            ),
            "tickers": ",".join(self.tickers),
            "exchanges": ",".join(self.exchanges),
            "firstFilingDate": self.first_filing_date or "",
            "lastFilingDate": self.last_filing_date or "",
            "first10KDate": ten_k_dates[0] if ten_k_dates else "",
            "last10KDate": ten_k_dates[-1] if ten_k_dates else "",
            "numberOf10Ks": len(ten_k_dates),
            "firstItem103Date": (
                item_103_dates[0] if item_103_dates else ""
            ),
            "lastItem103Date": (
                item_103_dates[-1] if item_103_dates else ""
            ),
            "numberOfItem103Filings": len(item_103_dates),
            "hasDistressEvent": int(bool(item_103_dates)),
        }


OUTPUT_COLUMNS = [
    "cik",
    "company",
    "entityType",
    "sic",
    "sicDescription",
    "fiscalYearEnd",
    "stateOfIncorporation",
    "tickers",
    "exchanges",
    "firstFilingDate",
    "lastFilingDate",
    "first10KDate",
    "last10KDate",
    "numberOf10Ks",
    "firstItem103Date",
    "lastItem103Date",
    "numberOfItem103Filings",
    "hasDistressEvent",
]


def clean_text(value: object) -> str:
    """Return a stripped string, treating null-like values as blank."""
    if value is None:
        return ""

    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def unique_strings(values: Iterable[object]) -> list[str]:
    """Return unique nonblank strings while preserving order."""
    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)

    return result


def normalize_cik(value: object) -> str:
    """Return a 10-digit CIK."""
    text = clean_text(value)

    if not text.isdigit():
        raise ValueError(f"Invalid CIK: {value!r}")

    if len(text) > 10:
        raise ValueError(f"CIK is longer than 10 digits: {text}")

    return text.zfill(10)


def normalize_iso_date(value: object) -> str:
    """Validate and return an ISO YYYY-MM-DD date, or blank."""
    text = clean_text(value)

    if not text:
        return ""

    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return ""


def cik_from_member_name(member_name: str) -> str | None:
    """Extract a CIK from a ZIP member name."""
    match = CIK_PATTERN.search(Path(member_name).name)
    return match.group(1) if match else None


def filing_columns(
    payload: dict[str, Any],
) -> dict[str, list[Any]] | None:
    """Find the column-oriented filing arrays in an SEC JSON object."""
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


def iter_relevant_filings(
    columns: dict[str, list[Any]],
) -> Iterable[tuple[object, object, object, object]]:
    """Yield accession, filing date, form, and items values."""
    accessions = columns.get("accessionNumber", [])
    filing_dates = columns.get("filingDate", [])
    forms = columns.get("form", [])
    items_values = columns.get("items", [])

    for accession, filing_date, form, items in zip_longest(
        accessions,
        filing_dates,
        forms,
        items_values,
        fillvalue="",
    ):
        yield accession, filing_date, form, items


def scan_archive(
    archive_path: Path,
    progress_every: int = 5_000,
) -> tuple[dict[str, FilerSummary], dict[str, int]]:
    """Scan the SEC submissions ZIP without extracting it."""
    summaries: dict[str, FilerSummary] = {}
    stats = {
        "jsonMembers": 0,
        "mainFiles": 0,
        "historicalFiles": 0,
        "membersWithoutCik": 0,
        "invalidJson": 0,
        "membersWithoutFilings": 0,
    }

    with zipfile.ZipFile(archive_path) as archive:
        json_members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and member.filename.lower().endswith(".json")
        ]

        total = len(json_members)

        for index, member in enumerate(json_members, start=1):
            stats["jsonMembers"] += 1
            cik_from_name = cik_from_member_name(member.filename)

            try:
                with archive.open(member) as source:
                    payload = json.load(source)
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
            ):
                stats["invalidJson"] += 1
                continue

            if not isinstance(payload, dict):
                stats["invalidJson"] += 1
                continue

            payload_cik = payload.get("cik")
            try:
                cik = (
                    normalize_cik(payload_cik)
                    if payload_cik is not None
                    else cik_from_name
                )
            except ValueError:
                cik = cik_from_name

            if not cik:
                stats["membersWithoutCik"] += 1
                continue

            summary = summaries.setdefault(
                cik,
                FilerSummary(cik=cik),
            )

            if "filings" in payload:
                stats["mainFiles"] += 1
                summary.update_metadata(payload)
            else:
                stats["historicalFiles"] += 1

            columns = filing_columns(payload)

            if columns is None:
                stats["membersWithoutFilings"] += 1
                continue

            for filing in iter_relevant_filings(columns):
                summary.add_filing(*filing)

            if progress_every > 0 and index % progress_every == 0:
                print(
                    f"\rScanned {index:,}/{total:,} JSON members; "
                    f"{len(summaries):,} filers",
                    end="",
                    flush=True,
                )

        if total >= progress_every > 0:
            print()

    return summaries, stats


def write_rows(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Write rows to CSV with stable column ordering."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=OUTPUT_COLUMNS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def select_model_candidates(
    rows: list[dict[str, object]],
    minimum_10ks: int,
    analysis_start: str,
) -> list[dict[str, object]]:
    """Select filers with enough annual-report history for a pilot."""
    start_date = normalize_iso_date(analysis_start)

    if not start_date:
        raise ValueError(
            "--analysis-start must be a valid YYYY-MM-DD date."
        )

    return [
        row
        for row in rows
        if int(row["numberOf10Ks"]) >= minimum_10ks
        and clean_text(row["last10KDate"]) >= start_date
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a historical filer universe from the SEC bulk "
            "submissions archive."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/raw/sec_bulk/submissions.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/universe/filer_universe.csv"
        ),
    )
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=Path(
            "data/processed/universe/model_candidates.csv"
        ),
    )
    parser.add_argument(
        "--min-10ks",
        type=int,
        default=3,
        help="Minimum number of original 10-K filings. Default: 3.",
    )
    parser.add_argument(
        "--analysis-start",
        default="2009-01-01",
        help=(
            "Require at least one 10-K on or after this date. "
            "Default: 2009-01-01."
        ),
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

    if args.min_10ks < 1:
        print(
            "Error: --min-10ks must be at least 1.",
            file=sys.stderr,
        )
        return 1

    try:
        summaries, stats = scan_archive(
            archive_path=args.archive,
            progress_every=args.progress_every,
        )

        rows = [
            summary.to_row()
            for summary in summaries.values()
        ]
        rows.sort(
            key=lambda row: (
                clean_text(row["company"]).upper(),
                clean_text(row["cik"]),
            )
        )

        candidates = select_model_candidates(
            rows=rows,
            minimum_10ks=args.min_10ks,
            analysis_start=args.analysis_start,
        )

        write_rows(rows, args.output)
        write_rows(candidates, args.candidates_output)

    except (
        FileNotFoundError,
        OSError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    ten_k_filers = sum(
        int(row["numberOf10Ks"]) > 0
        for row in rows
    )
    event_filers = sum(
        int(row["hasDistressEvent"]) == 1
        for row in rows
    )
    candidate_events = sum(
        int(row["hasDistressEvent"]) == 1
        for row in candidates
    )

    print(f"JSON members scanned: {stats['jsonMembers']:,}")
    print(f"Filers found: {len(rows):,}")
    print(f"Filers with at least one 10-K: {ten_k_filers:,}")
    print(f"Filers with Item 1.03 disclosure: {event_filers:,}")
    print(
        f"Model candidates: {len(candidates):,} "
        f"(minimum {args.min_10ks} 10-Ks; "
        f"last 10-K on/after {args.analysis_start})"
    )
    print(
        "Candidates with Item 1.03 disclosure: "
        f"{candidate_events:,}"
    )
    print(f"Full universe: {args.output}")
    print(f"Candidate universe: {args.candidates_output}")

    if stats["invalidJson"]:
        print(
            f"Warning: invalid JSON members skipped: "
            f"{stats['invalidJson']:,}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
