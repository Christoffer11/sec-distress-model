"""Build the neutral SEC event-disclosure table.

The script scans the complete SEC ``submissions.zip`` archive and extracts
every Form 8-K or 8-K/A filing whose SEC ``items`` field contains Item 1.03
(Bankruptcy or Receivership).

This is a disclosure table, not a model target table:
- repeated Item 1.03 filings for the same CIK are retained;
- no attempt is made to identify a first distress onset;
- no company, industry, date, or sample filters are applied;
- the filing date is stored as the disclosure date, not asserted to be the
  underlying bankruptcy petition date.

Default output
--------------
data/base/events.parquet

Example
-------
PowerShell:
    python src/base_tables/build_events.py

Optional CSV output:
    python src/base_tables/build_events.py `
        --output data/base/events.csv
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


CIK_PATTERN = re.compile(r"CIK(\d{1,10})", flags=re.IGNORECASE)
ITEM_NUMBER_PATTERN = re.compile(r"(?<!\d)(\d+\.\d{2})(?!\d)")

QUALIFYING_FORMS = {"8-K", "8-K/A"}
TARGET_ITEM = "1.03"

DEFAULT_ARCHIVE = Path("data/raw/sec_bulk/submissions.zip")
DEFAULT_OUTPUT = Path("data/base/events.parquet")

SEC_ARCHIVES_ROOT = "https://www.sec.gov/Archives/edgar/data"

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

    # Supplementary history files store filing arrays at the top level.
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


def parse_item_numbers(value: object) -> list[str]:
    """Extract normalized SEC item numbers while preserving order."""
    text = clean_text(value)

    if text is None:
        return []

    result: list[str] = []
    seen: set[str] = set()

    for match in ITEM_NUMBER_PATTERN.finditer(text):
        item_number = match.group(1)

        if item_number not in seen:
            seen.add(item_number)
            result.append(item_number)

    return result


def iter_filing_rows(
    columns: dict[str, list[Any]],
) -> Iterable[dict[str, object]]:
    """Yield rows from the SEC's column-oriented filing arrays."""
    arrays = [
        columns.get(field, [])
        for field in FILING_FIELDS
    ]

    for values in zip_longest(*arrays, fillvalue=None):
        yield dict(zip(FILING_FIELDS, values))


def build_filing_url(
    cik: str,
    accession_number: str | None,
    primary_document: str | None,
) -> str | None:
    """Build the direct SEC URL for the filing's primary document."""
    if accession_number is None or primary_document is None:
        return None

    accession_folder = accession_number.replace("-", "")

    if not accession_folder:
        return None

    cik_without_leading_zeros = str(int(cik))

    return (
        f"{SEC_ARCHIVES_ROOT}/"
        f"{cik_without_leading_zeros}/"
        f"{accession_folder}/"
        f"{primary_document}"
    )


def scan_events_archive(
    archive_path: Path,
    progress_every: int = 5_000,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Scan the complete submissions archive for Item 1.03 filings."""
    rows: list[dict[str, object]] = []

    stats = {
        "json_members": 0,
        "main_records": 0,
        "supplementary_records": 0,
        "invalid_json": 0,
        "members_without_cik": 0,
        "members_without_filings": 0,
        "qualifying_rows_before_deduplication": 0,
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

            if isinstance(payload.get("filings"), dict):
                stats["main_records"] += 1
            else:
                stats["supplementary_records"] += 1

            columns = filing_columns(payload)

            if columns is None:
                stats["members_without_filings"] += 1
                continue

            for filing in iter_filing_rows(columns):
                form = (clean_text(filing.get("form")) or "").upper()

                if form not in QUALIFYING_FORMS:
                    continue

                items_raw = clean_text(filing.get("items"))
                item_numbers = parse_item_numbers(items_raw)

                if TARGET_ITEM not in item_numbers:
                    continue

                accession_number = clean_text(
                    filing.get("accessionNumber")
                )
                primary_document = clean_text(
                    filing.get("primaryDocument")
                )

                rows.append(
                    {
                        "event_id": (
                            f"{cik}:"
                            f"{accession_number or 'missing'}:"
                            f"{TARGET_ITEM}"
                        ),
                        "cik": cik,
                        "event_type": (
                            "bankruptcy_or_receivership_disclosure"
                        ),
                        "event_code": "8-K_ITEM_1.03",
                        "disclosure_date": clean_text(
                            filing.get("filingDate")
                        ),
                        "report_date": clean_text(
                            filing.get("reportDate")
                        ),
                        "acceptance_datetime": clean_text(
                            filing.get("acceptanceDateTime")
                        ),
                        "form": form,
                        "items_raw": items_raw,
                        "item_numbers": item_numbers,
                        "accession_number": accession_number,
                        "act": clean_text(filing.get("act")),
                        "file_number": clean_text(
                            filing.get("fileNumber")
                        ),
                        "film_number": clean_text(
                            filing.get("filmNumber")
                        ),
                        "filing_size_bytes": filing.get("size"),
                        "is_xbrl": filing.get("isXBRL"),
                        "is_inline_xbrl": filing.get(
                            "isInlineXBRL"
                        ),
                        "primary_document": primary_document,
                        "primary_document_description": clean_text(
                            filing.get("primaryDocDescription")
                        ),
                        "filing_url": build_filing_url(
                            cik=cik,
                            accession_number=accession_number,
                            primary_document=primary_document,
                        ),
                        "source_member": member.filename,
                    }
                )
                stats[
                    "qualifying_rows_before_deduplication"
                ] += 1

            if (
                progress_every > 0
                and index % progress_every == 0
            ):
                print(
                    f"\rScanned {index:,}/{total:,} JSON members; "
                    f"{len(rows):,} qualifying disclosures",
                    end="",
                    flush=True,
                )

        if progress_every > 0 and total >= progress_every:
            print()

    columns = [
        "event_id",
        "cik",
        "event_type",
        "event_code",
        "disclosure_date",
        "report_date",
        "acceptance_datetime",
        "form",
        "items_raw",
        "item_numbers",
        "accession_number",
        "act",
        "file_number",
        "film_number",
        "filing_size_bytes",
        "is_xbrl",
        "is_inline_xbrl",
        "primary_document",
        "primary_document_description",
        "filing_url",
        "source_member",
    ]

    events = pd.DataFrame(rows, columns=columns)

    if events.empty:
        stats["exact_duplicates_removed"] = 0
        return events, stats

    before = len(events)

    # This removes only duplicate representations of the same SEC filing.
    # Repeated disclosures with different accession numbers remain.
    events = (
        events.sort_values(
            [
                "cik",
                "disclosure_date",
                "accession_number",
                "source_member",
            ],
            na_position="last",
        )
        .drop_duplicates(
            subset=[
                "cik",
                "accession_number",
                "event_code",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    stats["exact_duplicates_removed"] = before - len(events)

    for column in ("disclosure_date", "report_date"):
        events[column] = pd.to_datetime(
            events[column],
            errors="coerce",
        )

    events["acceptance_datetime"] = pd.to_datetime(
        events["acceptance_datetime"],
        errors="coerce",
        utc=True,
    )

    for column in (
        "filing_size_bytes",
        "is_xbrl",
        "is_inline_xbrl",
    ):
        events[column] = pd.to_numeric(
            events[column],
            errors="coerce",
        ).astype("Int64")

    return events, stats


def write_table(
    events: pd.DataFrame,
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
            events.to_parquet(
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
        csv_events = events.copy()
        csv_events["item_numbers"] = csv_events[
            "item_numbers"
        ].map(lambda values: "|".join(values))
        csv_events.to_csv(
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
            "Build one neutral event row per SEC 8-K Item 1.03 "
            "disclosure."
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
        events, stats = scan_events_archive(
            archive_path=args.archive,
            progress_every=args.progress_every,
        )
        write_table(
            events=events,
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

    print(f"Item 1.03 disclosures: {len(events):,}")
    print(
        "Unique CIKs with a disclosure: "
        f"{events['cik'].nunique():,}"
    )
    print(
        "Exact duplicate archive rows removed: "
        f"{stats['exact_duplicates_removed']:,}"
    )

    if not events.empty:
        print(
            "First disclosure date: "
            f"{events['disclosure_date'].min().date()}"
        )
        print(
            "Last disclosure date: "
            f"{events['disclosure_date'].max().date()}"
        )
        print("Forms:")
        print(
            events["form"]
            .value_counts(dropna=False)
            .to_string()
        )

    print(f"Invalid JSON members skipped: {stats['invalid_json']:,}")
    print(f"Saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
