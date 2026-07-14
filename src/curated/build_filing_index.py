"""Build a filing-level index from the SEC bulk submissions archive.

The output has one row per CIK-accession combination and is intended to join
filing metadata to the long-form Company Facts table.

No modelling filters are applied by default. All filing forms are retained.
Use ``--forms`` only when deliberately building a narrower index.

Default output
--------------
data/curated/filing_index.parquet

Examples
--------
Build the complete filing index:

    python src/curated/build_filing_index.py

Build only annual filing records:

    python src/curated/build_filing_index.py \
        --forms 10-K 10-K/A \
        --output data/curated/annual_filing_index.parquet

Rebuild an existing output:

    python src/curated/build_filing_index.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import date, datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


CIK_PATTERN = re.compile(r"CIK(\d{1,10})", flags=re.IGNORECASE)

DEFAULT_ARCHIVE = Path("data/raw/sec_bulk/submissions.zip")
DEFAULT_OUTPUT = Path("data/curated/filing_index.parquet")
DEFAULT_TEMP_DIR = Path("data/curated/_filing_index_parts")
DEFAULT_BATCH_ROWS = 200_000

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

FILING_SCHEMA = pa.schema(
    [
        pa.field("cik", pa.string(), nullable=False),
        pa.field("accession_number", pa.string()),
        pa.field("filing_date", pa.date32()),
        pa.field("report_date", pa.date32()),
        pa.field("acceptance_datetime", pa.timestamp("us", tz="UTC")),
        pa.field("act", pa.string()),
        pa.field("form", pa.string()),
        pa.field("file_number", pa.string()),
        pa.field("film_number", pa.string()),
        pa.field("items_raw", pa.string()),
        pa.field("filing_size_bytes", pa.int64()),
        pa.field("is_xbrl", pa.int8()),
        pa.field("is_inline_xbrl", pa.int8()),
        pa.field("primary_document", pa.string()),
        pa.field("primary_document_description", pa.string()),
        pa.field("filing_url", pa.string()),
        pa.field("source_member", pa.string(), nullable=False),
        pa.field("source_row_index", pa.int32(), nullable=False),
        pa.field("source_is_supplementary", pa.bool_(), nullable=False),
    ]
)

SEC_ARCHIVES_ROOT = "https://www.sec.gov/Archives/edgar/data"


def clean_text(value: object) -> str | None:
    """Return a stripped nullable string."""
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.casefold() in {"none", "null", "nan"}:
        return None

    return text


def normalize_cik(value: object) -> str:
    """Return a zero-padded 10-digit CIK."""
    text = clean_text(value)

    if text is None or not text.isdigit():
        raise ValueError(f"Invalid CIK: {value!r}")

    if len(text) > 10:
        raise ValueError(f"CIK is longer than 10 digits: {text}")

    return text.zfill(10)


def cik_from_member_name(member_name: str) -> str | None:
    """Extract the filer CIK from a submissions member filename."""
    match = CIK_PATTERN.search(Path(member_name).name)

    if match is None:
        return None

    return match.group(1).zfill(10)


def parse_date(value: object) -> date | None:
    """Parse an ISO date, returning None when unavailable."""
    text = clean_text(value)

    if text is None:
        return None

    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def parse_datetime_utc(value: object) -> datetime | None:
    """Parse an SEC acceptance datetime and normalize it to UTC."""
    text = clean_text(value)

    if text is None:
        return None

    normalized = text.replace("Z", "+00:00")

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    return parsed


def parse_integer(value: object) -> int | None:
    """Parse an integer-like field without raising."""
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def parse_binary_flag(value: object) -> int | None:
    """Parse SEC zero/one flags to nullable integers."""
    parsed = parse_integer(value)

    if parsed in {0, 1}:
        return parsed

    return None


def filing_columns(payload: dict[str, Any]) -> dict[str, list[Any]] | None:
    """Locate filing arrays in current or supplementary JSON."""
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


def iter_filing_records(
    columns: dict[str, list[Any]],
) -> Iterable[tuple[int, dict[str, object]]]:
    """Yield row positions and filing records from column arrays."""
    arrays = [columns.get(field, []) for field in FILING_FIELDS]

    for index, values in enumerate(zip_longest(*arrays, fillvalue=None)):
        yield index, dict(zip(FILING_FIELDS, values))


def build_filing_url(
    cik: str,
    accession_number: str | None,
    primary_document: str | None,
) -> str | None:
    """Build the direct SEC URL for the primary filing document."""
    if accession_number is None or primary_document is None:
        return None

    accession_folder = accession_number.replace("-", "")

    if not accession_folder:
        return None

    return (
        f"{SEC_ARCHIVES_ROOT}/"
        f"{int(cik)}/"
        f"{accession_folder}/"
        f"{primary_document}"
    )


def row_from_filing(
    cik: str,
    filing: dict[str, object],
    source_member: str,
    source_row_index: int,
) -> dict[str, object]:
    """Convert one raw SEC filing record to the index schema."""
    accession_number = clean_text(filing.get("accessionNumber"))
    primary_document = clean_text(filing.get("primaryDocument"))

    return {
        "cik": cik,
        "accession_number": accession_number,
        "filing_date": parse_date(filing.get("filingDate")),
        "report_date": parse_date(filing.get("reportDate")),
        "acceptance_datetime": parse_datetime_utc(
            filing.get("acceptanceDateTime")
        ),
        "act": clean_text(filing.get("act")),
        "form": clean_text(filing.get("form")),
        "file_number": clean_text(filing.get("fileNumber")),
        "film_number": clean_text(filing.get("filmNumber")),
        "items_raw": clean_text(filing.get("items")),
        "filing_size_bytes": parse_integer(filing.get("size")),
        "is_xbrl": parse_binary_flag(filing.get("isXBRL")),
        "is_inline_xbrl": parse_binary_flag(
            filing.get("isInlineXBRL")
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
        "source_member": source_member,
        "source_row_index": source_row_index,
        "source_is_supplementary": (
            "-submissions-" in source_member.casefold()
        ),
    }


def write_part(
    rows: list[dict[str, object]],
    temp_dir: Path,
    part_number: int,
) -> Path:
    """Write one temporary Parquet part atomically."""
    temp_dir.mkdir(parents=True, exist_ok=True)

    destination = temp_dir / f"part-{part_number:05d}.parquet"
    temporary = temp_dir / f".part-{part_number:05d}.tmp"

    table = pa.Table.from_pylist(rows, schema=FILING_SCHEMA)

    pq.write_table(
        table,
        temporary,
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, destination)

    return destination


def sql_path(path: Path) -> str:
    """Return a DuckDB-safe POSIX path literal."""
    return path.resolve().as_posix().replace("'", "''")


def consolidate_parts(temp_dir: Path, output_path: Path) -> dict[str, int]:
    """Deduplicate temporary parts and write the final index."""
    part_glob = sql_path(temp_dir / "*.parquet")
    output_sql_path = sql_path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    con = duckdb.connect()

    try:
        raw_rows = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{part_glob}')
            """
        ).fetchone()[0]

        con.execute(
            f"""
            COPY (
                WITH ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY
                                cik,
                                COALESCE(
                                    accession_number,
                                    source_member
                                    || '#'
                                    || CAST(source_row_index AS VARCHAR)
                                )
                            ORDER BY
                                source_is_supplementary ASC,
                                (
                                    CAST(report_date IS NOT NULL AS INTEGER)
                                    + CAST(
                                        acceptance_datetime IS NOT NULL
                                        AS INTEGER
                                    )
                                    + CAST(
                                        primary_document IS NOT NULL
                                        AS INTEGER
                                    )
                                    + CAST(
                                        filing_size_bytes IS NOT NULL
                                        AS INTEGER
                                    )
                                ) DESC,
                                source_member,
                                source_row_index
                        ) AS duplicate_rank
                    FROM read_parquet('{part_glob}')
                )
                SELECT * EXCLUDE (duplicate_rank)
                FROM ranked
                WHERE duplicate_rank = 1
                ORDER BY cik, filing_date, accession_number
            )
            TO '{output_sql_path}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 250000
            )
            """
        )

        final_rows = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{output_sql_path}')
            """
        ).fetchone()[0]

        unique_ciks = con.execute(
            f"""
            SELECT COUNT(DISTINCT cik)
            FROM read_parquet('{output_sql_path}')
            """
        ).fetchone()[0]

        missing_accessions = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{output_sql_path}')
            WHERE accession_number IS NULL
            """
        ).fetchone()[0]

    finally:
        con.close()

    return {
        "raw_rows": int(raw_rows),
        "final_rows": int(final_rows),
        "duplicates_removed": int(raw_rows - final_rows),
        "unique_ciks": int(unique_ciks),
        "missing_accessions": int(missing_accessions),
    }


def build_filing_index(
    archive_path: Path,
    output_path: Path,
    temp_dir: Path,
    forms: list[str] | None,
    batch_rows: int,
    progress_every: int,
    overwrite: bool,
    keep_temp: bool,
) -> dict[str, int]:
    """Extract, deduplicate, and write the filing index."""
    if batch_rows < 1:
        raise ValueError("--batch-rows must be at least 1.")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Use --overwrite to rebuild it."
        )

    if overwrite:
        output_path.unlink(missing_ok=True)

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    elif temp_dir.exists():
        raise FileExistsError(
            f"Temporary directory already exists: {temp_dir}. "
            "Use --overwrite to restart cleanly."
        )

    wanted_forms = (
        {form.strip().upper() for form in forms}
        if forms
        else None
    )

    rows: list[dict[str, object]] = []
    part_number = 0

    stats = {
        "archive_members": 0,
        "members_with_filings": 0,
        "members_without_filings": 0,
        "invalid_json": 0,
        "members_without_cik": 0,
        "rows_extracted": 0,
        "parts_written": 0,
    }

    with zipfile.ZipFile(archive_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and member.filename.lower().endswith(".json")
        ]

        stats["archive_members"] = len(members)

        for member_index, member in enumerate(members, start=1):
            try:
                with archive.open(member) as source:
                    payload = json.load(source)
            except (json.JSONDecodeError, UnicodeDecodeError):
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

            columns = filing_columns(payload)

            if columns is None:
                stats["members_without_filings"] += 1
                continue

            stats["members_with_filings"] += 1

            for source_row_index, filing in iter_filing_records(columns):
                form = (clean_text(filing.get("form")) or "").upper()

                if wanted_forms is not None and form not in wanted_forms:
                    continue

                rows.append(
                    row_from_filing(
                        cik=cik,
                        filing=filing,
                        source_member=member.filename,
                        source_row_index=source_row_index,
                    )
                )
                stats["rows_extracted"] += 1

                if len(rows) >= batch_rows:
                    write_part(
                        rows=rows,
                        temp_dir=temp_dir,
                        part_number=part_number,
                    )
                    rows = []
                    part_number += 1
                    stats["parts_written"] += 1

            if (
                progress_every > 0
                and member_index % progress_every == 0
            ):
                print(
                    f"Processed {member_index:,}/{len(members):,} "
                    f"archive members; {stats['rows_extracted']:,} "
                    "filing rows"
                )

        if rows:
            write_part(
                rows=rows,
                temp_dir=temp_dir,
                part_number=part_number,
            )
            stats["parts_written"] += 1

    if stats["rows_extracted"] == 0:
        raise RuntimeError("No filing rows were extracted.")

    consolidation = consolidate_parts(
        temp_dir=temp_dir,
        output_path=output_path,
    )
    stats.update(consolidation)

    if not keep_temp:
        shutil.rmtree(temp_dir)

    return stats


def print_form_summary(output_path: Path) -> None:
    """Print the largest filing-form categories."""
    path = sql_path(output_path)
    con = duckdb.connect()

    try:
        result = con.execute(
            f"""
            SELECT
                form,
                COUNT(*) AS filings,
                COUNT(DISTINCT cik) AS filers
            FROM read_parquet('{path}')
            GROUP BY form
            ORDER BY filings DESC
            LIMIT 20
            """
        ).fetchall()
    finally:
        con.close()

    print("Top forms:")

    for form, filings, filers in result:
        print(
            f"  {str(form):<12} "
            f"{filings:>12,} filings "
            f"{filers:>12,} filers"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one filing-index row per SEC CIK-accession."
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
        help=f"Output Parquet file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=DEFAULT_TEMP_DIR,
        help=(
            "Temporary Parquet-part directory. "
            f"Default: {DEFAULT_TEMP_DIR}"
        ),
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        help=(
            "Optional form filter, e.g. --forms 10-K 10-K/A. "
            "Omit to retain every form."
        ),
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=DEFAULT_BATCH_ROWS,
        help=(
            "Rows per temporary Parquet part. "
            f"Default: {DEFAULT_BATCH_ROWS:,}."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25_000,
        help=(
            "Print progress after this many JSON members. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and temporary files.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary Parquet parts after consolidation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.output.suffix.casefold() != ".parquet":
        print("Error: --output must end in .parquet", file=sys.stderr)
        return 1

    try:
        stats = build_filing_index(
            archive_path=args.archive,
            output_path=args.output,
            temp_dir=args.temp_dir,
            forms=args.forms,
            batch_rows=args.batch_rows,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
            keep_temp=args.keep_temp,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"Archive members: {stats['archive_members']:,}")
    print(
        "Members with filing arrays: "
        f"{stats['members_with_filings']:,}"
    )
    print(
        "Members without filing arrays: "
        f"{stats['members_without_filings']:,}"
    )
    print(f"Invalid JSON members: {stats['invalid_json']:,}")
    print(f"Raw filing rows: {stats['raw_rows']:,}")
    print(f"Final filing rows: {stats['final_rows']:,}")
    print(
        "Duplicate archive representations removed: "
        f"{stats['duplicates_removed']:,}"
    )
    print(f"Unique CIKs: {stats['unique_ciks']:,}")
    print(
        "Rows without accession number: "
        f"{stats['missing_accessions']:,}"
    )
    print(f"Saved to: {args.output}")
    print()

    print_form_summary(args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
