"""Build the neutral SEC financial-facts base table.

The script scans the complete SEC ``companyfacts.zip`` archive and writes one
long-form row per reported Company Facts observation.

No modelling decisions are applied:
- all available taxonomies are retained;
- all concepts and units are retained;
- all forms and reporting periods are retained;
- comparative and subsequently re-filed observations are retained;
- no preferred concept mappings or financial ratios are created.

Because the complete table can be very large, it is written as a partitioned
Parquet dataset:

    data/base/financial_facts/
        part-00000.parquet
        part-00001.parquet
        ...

Supporting extraction metadata is written separately:

    data/base/financial_facts_manifest.parquet
    data/base/financial_facts_failures.csv

Example
-------
PowerShell:
    python src/base_tables/build_financial_facts.py

Restart after interruption:
    python src/base_tables/build_financial_facts.py

Rebuild from scratch:
    python src/base_tables/build_financial_facts.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


CIK_PATTERN = re.compile(r"CIK(\d{1,10})", flags=re.IGNORECASE)

DEFAULT_ARCHIVE = Path("data/raw/sec_bulk/companyfacts.zip")
DEFAULT_OUTPUT_DIR = Path("data/base/financial_facts")
DEFAULT_MANIFEST = Path(
    "data/base/financial_facts_manifest.parquet"
)
DEFAULT_FAILURES = Path(
    "data/base/financial_facts_failures.csv"
)

DEFAULT_MAX_ROWS_PER_PART = 250_000


FACT_SCHEMA = pa.schema(
    [
        pa.field("cik", pa.string(), nullable=False),
        pa.field("entity_name", pa.string()),
        pa.field("taxonomy", pa.string(), nullable=False),
        pa.field("concept", pa.string(), nullable=False),
        pa.field("label", pa.string()),
        pa.field("description", pa.string()),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("value_text", pa.string()),
        pa.field("value_numeric", pa.float64()),
        pa.field("value_type", pa.string()),
        pa.field("start_date", pa.date32()),
        pa.field("end_date", pa.date32()),
        pa.field("accession_number", pa.string()),
        pa.field("fiscal_year", pa.int32()),
        pa.field("fiscal_period", pa.string()),
        pa.field("form", pa.string()),
        pa.field("filed_date", pa.date32()),
        pa.field("frame", pa.string()),
        pa.field("source_member", pa.string(), nullable=False),
        pa.field("source_observation_index", pa.int32(), nullable=False),
    ]
)


MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("source_member", pa.string(), nullable=False),
        pa.field("cik", pa.string()),
        pa.field("entity_name", pa.string()),
        pa.field("status", pa.string(), nullable=False),
        pa.field("fact_rows", pa.int64(), nullable=False),
        pa.field("part_file", pa.string()),
        pa.field("error", pa.string()),
    ]
)


def clean_text(value: object) -> str | None:
    """Convert a value to a clean nullable string."""
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
    """Extract a CIK from a Company Facts JSON filename."""
    match = CIK_PATTERN.search(Path(member_name).name)

    if match is None:
        return None

    return match.group(1).zfill(10)


def parse_date(value: object) -> date | None:
    """Parse an ISO date value to ``datetime.date``."""
    text = clean_text(value)

    if text is None:
        return None

    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def parse_int(value: object) -> int | None:
    """Parse an integer-like SEC field."""
    if value is None or isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def value_fields(value: object) -> tuple[str | None, float | None, str]:
    """Return a loss-aware textual value, numeric convenience value, and type."""
    if value is None:
        return None, None, "null"

    if isinstance(value, bool):
        return ("true" if value else "false"), None, "boolean"

    if isinstance(value, int):
        numeric = float(value)

        if not math.isfinite(numeric):
            numeric = None

        return str(value), numeric, "integer"

    if isinstance(value, float):
        numeric = value if math.isfinite(value) else None
        return repr(value), numeric, "number"

    if isinstance(value, str):
        return value, None, "string"

    # Preserve unusual JSON-compatible values rather than discarding them.
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ), None, type(value).__name__
    except TypeError:
        return str(value), None, type(value).__name__


def iter_fact_rows(
    payload: dict[str, Any],
    source_member: str,
) -> Iterable[dict[str, object]]:
    """Yield one long-form row per Company Facts observation."""
    cik_value = payload.get("cik")

    try:
        cik = normalize_cik(cik_value)
    except ValueError:
        cik_from_name = cik_from_member_name(source_member)

        if cik_from_name is None:
            raise

        cik = cik_from_name

    entity_name = clean_text(payload.get("entityName"))
    facts = payload.get("facts")

    if not isinstance(facts, dict):
        return

    observation_index = 0

    for taxonomy, concepts in facts.items():
        if not isinstance(concepts, dict):
            continue

        taxonomy_text = clean_text(taxonomy)

        if taxonomy_text is None:
            continue

        for concept, concept_payload in concepts.items():
            if not isinstance(concept_payload, dict):
                continue

            concept_text = clean_text(concept)

            if concept_text is None:
                continue

            label = clean_text(concept_payload.get("label"))
            description = clean_text(
                concept_payload.get("description")
            )
            units = concept_payload.get("units")

            if not isinstance(units, dict):
                continue

            for unit, observations in units.items():
                unit_text = clean_text(unit)

                if unit_text is None or not isinstance(
                    observations,
                    list,
                ):
                    continue

                for observation in observations:
                    if not isinstance(observation, dict):
                        continue

                    value_text, value_numeric, value_type = (
                        value_fields(observation.get("val"))
                    )

                    yield {
                        "cik": cik,
                        "entity_name": entity_name,
                        "taxonomy": taxonomy_text,
                        "concept": concept_text,
                        "label": label,
                        "description": description,
                        "unit": unit_text,
                        "value_text": value_text,
                        "value_numeric": value_numeric,
                        "value_type": value_type,
                        "start_date": parse_date(
                            observation.get("start")
                        ),
                        "end_date": parse_date(
                            observation.get("end")
                        ),
                        "accession_number": clean_text(
                            observation.get("accn")
                        ),
                        "fiscal_year": parse_int(
                            observation.get("fy")
                        ),
                        "fiscal_period": clean_text(
                            observation.get("fp")
                        ),
                        "form": clean_text(
                            observation.get("form")
                        ),
                        "filed_date": parse_date(
                            observation.get("filed")
                        ),
                        "frame": clean_text(
                            observation.get("frame")
                        ),
                        "source_member": source_member,
                        "source_observation_index": (
                            observation_index
                        ),
                    }
                    observation_index += 1


def list_json_members(
    archive: zipfile.ZipFile,
) -> list[zipfile.ZipInfo]:
    """Return sorted JSON members from the bulk archive."""
    return sorted(
        [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and member.filename.lower().endswith(".json")
        ],
        key=lambda member: member.filename,
    )


def read_manifest(path: Path) -> list[dict[str, object]]:
    """Read a prior extraction manifest when resuming."""
    if not path.exists():
        return []

    table = pq.read_table(path)
    return table.to_pylist()


def write_manifest(
    records: list[dict[str, object]],
    path: Path,
) -> None:
    """Atomically write the extraction manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    table = pa.Table.from_pylist(
        records,
        schema=MANIFEST_SCHEMA,
    )
    pq.write_table(
        table,
        temporary,
        compression="snappy",
    )
    os.replace(temporary, path)


def write_failures(
    records: list[dict[str, object]],
    path: Path,
) -> None:
    """Write failures in a human-readable CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_member",
        "cik",
        "entity_name",
        "status",
        "fact_rows",
        "part_file",
        "error",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(records)


def existing_part_numbers(output_dir: Path) -> list[int]:
    """Return numeric suffixes from existing part files."""
    numbers: list[int] = []

    for path in output_dir.glob("part-*.parquet"):
        stem = path.stem
        suffix = stem.removeprefix("part-")

        if suffix.isdigit():
            numbers.append(int(suffix))

    return sorted(numbers)


def validate_resume_state(
    output_dir: Path,
    manifest_records: list[dict[str, object]],
) -> int:
    """Validate part files against the manifest and return the next part ID."""
    existing_numbers = existing_part_numbers(output_dir)

    manifest_parts = {
        str(record["part_file"])
        for record in manifest_records
        if record.get("part_file")
    }
    existing_parts = {
        path.name
        for path in output_dir.glob("part-*.parquet")
    }

    orphan_parts = existing_parts - manifest_parts
    missing_parts = manifest_parts - existing_parts

    if orphan_parts:
        raise RuntimeError(
            "Part files exist but are absent from the manifest: "
            f"{sorted(orphan_parts)[:10]}. "
            "Use --overwrite to rebuild cleanly."
        )

    if missing_parts:
        raise RuntimeError(
            "The manifest references missing part files: "
            f"{sorted(missing_parts)[:10]}. "
            "Restore them or use --overwrite."
        )

    return max(existing_numbers, default=-1) + 1


def write_part(
    rows: list[dict[str, object]],
    output_dir: Path,
    part_number: int,
) -> str:
    """Atomically write one Parquet part and return its filename."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"part-{part_number:05d}.parquet"
    destination = output_dir / filename
    temporary = output_dir / f".{filename}.tmp"

    table = pa.Table.from_pylist(
        rows,
        schema=FACT_SCHEMA,
    )

    pq.write_table(
        table,
        temporary,
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, destination)

    return filename


def rebuild_output(
    output_dir: Path,
    manifest_path: Path,
    failures_path: Path,
) -> None:
    """Remove prior generated outputs for a clean rebuild."""
    if output_dir.exists():
        shutil.rmtree(output_dir)

    manifest_path.unlink(missing_ok=True)
    failures_path.unlink(missing_ok=True)


def build_financial_facts(
    archive_path: Path,
    output_dir: Path,
    manifest_path: Path,
    failures_path: Path,
    max_rows_per_part: int,
    progress_every: int,
    overwrite: bool,
    retry_failures: bool,
) -> dict[str, int]:
    """Extract the complete Company Facts archive to Parquet parts."""
    if max_rows_per_part < 1:
        raise ValueError(
            "--max-rows-per-part must be at least 1."
        )

    if overwrite:
        rebuild_output(
            output_dir=output_dir,
            manifest_path=manifest_path,
            failures_path=failures_path,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_records = read_manifest(manifest_path)
    next_part_number = validate_resume_state(
        output_dir=output_dir,
        manifest_records=manifest_records,
    )

    processed_members = {
        str(record["source_member"])
        for record in manifest_records
        if record.get("status") in {"written", "empty"}
        or (
            record.get("status") == "failed"
            and not retry_failures
        )
    }

    # Retried failures replace their prior manifest entries.
    if retry_failures:
        manifest_records = [
            record
            for record in manifest_records
            if record.get("status") != "failed"
        ]

    pending_rows: list[dict[str, object]] = []
    pending_manifest: list[dict[str, object]] = []

    stats = {
        "archive_members": 0,
        "members_skipped": 0,
        "members_written": 0,
        "members_empty": 0,
        "members_failed": 0,
        "fact_rows_written": 0,
        "parts_written": 0,
    }

    def flush_pending() -> None:
        nonlocal next_part_number
        nonlocal pending_rows
        nonlocal pending_manifest
        nonlocal manifest_records

        if not pending_manifest:
            return

        part_filename: str | None = None

        if pending_rows:
            part_filename = write_part(
                rows=pending_rows,
                output_dir=output_dir,
                part_number=next_part_number,
            )
            next_part_number += 1
            stats["parts_written"] += 1
            stats["fact_rows_written"] += len(pending_rows)

        for record in pending_manifest:
            if record["status"] == "written":
                record["part_file"] = part_filename

        manifest_records.extend(pending_manifest)
        write_manifest(
            records=manifest_records,
            path=manifest_path,
        )

        failures = [
            record
            for record in manifest_records
            if record["status"] == "failed"
        ]

        if failures:
            write_failures(
                records=failures,
                path=failures_path,
            )
        else:
            failures_path.unlink(missing_ok=True)

        pending_rows = []
        pending_manifest = []

    with zipfile.ZipFile(archive_path) as archive:
        members = list_json_members(archive)
        stats["archive_members"] = len(members)

        for index, member in enumerate(members, start=1):
            if member.filename in processed_members:
                stats["members_skipped"] += 1
                continue

            cik = cik_from_member_name(member.filename)
            entity_name: str | None = None

            try:
                with archive.open(member) as source:
                    payload = json.load(source)

                if not isinstance(payload, dict):
                    raise ValueError(
                        "JSON payload is not an object."
                    )

                entity_name = clean_text(
                    payload.get("entityName")
                )

                member_rows = list(
                    iter_fact_rows(
                        payload=payload,
                        source_member=member.filename,
                    )
                )

                if member_rows:
                    pending_rows.extend(member_rows)
                    pending_manifest.append(
                        {
                            "source_member": member.filename,
                            "cik": (
                                member_rows[0]["cik"]
                                if member_rows
                                else cik
                            ),
                            "entity_name": entity_name,
                            "status": "written",
                            "fact_rows": len(member_rows),
                            "part_file": None,
                            "error": None,
                        }
                    )
                    stats["members_written"] += 1
                else:
                    pending_manifest.append(
                        {
                            "source_member": member.filename,
                            "cik": cik,
                            "entity_name": entity_name,
                            "status": "empty",
                            "fact_rows": 0,
                            "part_file": None,
                            "error": None,
                        }
                    )
                    stats["members_empty"] += 1

            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                pending_manifest.append(
                    {
                        "source_member": member.filename,
                        "cik": cik,
                        "entity_name": entity_name,
                        "status": "failed",
                        "fact_rows": 0,
                        "part_file": None,
                        "error": str(exc),
                    }
                )
                stats["members_failed"] += 1

            if (
                len(pending_rows) >= max_rows_per_part
                or (
                    not pending_rows
                    and len(pending_manifest) >= 500
                )
            ):
                flush_pending()

            if (
                progress_every > 0
                and index % progress_every == 0
            ):
                completed = (
                    stats["members_skipped"]
                    + stats["members_written"]
                    + stats["members_empty"]
                    + stats["members_failed"]
                )
                print(
                    f"Processed {completed:,}/{len(members):,} "
                    f"archive members; "
                    f"{stats['fact_rows_written'] + len(pending_rows):,} "
                    "fact rows encountered"
                )

        flush_pending()

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the complete neutral SEC Company Facts "
            "Parquet dataset."
        )
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Input Company Facts ZIP. Default: {DEFAULT_ARCHIVE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Parquet dataset directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Extraction manifest. Default: {DEFAULT_MANIFEST}",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=DEFAULT_FAILURES,
        help=f"Failure log. Default: {DEFAULT_FAILURES}",
    )
    parser.add_argument(
        "--max-rows-per-part",
        type=int,
        default=DEFAULT_MAX_ROWS_PER_PART,
        help=(
            "Approximate maximum rows accumulated before writing "
            f"a part. Default: {DEFAULT_MAX_ROWS_PER_PART:,}."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help=(
            "Print progress after this many archive members. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete generated outputs and rebuild from scratch.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry members previously recorded as failed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        stats = build_financial_facts(
            archive_path=args.archive,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            failures_path=args.failures,
            max_rows_per_part=args.max_rows_per_part,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
            retry_failures=args.retry_failures,
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

    print()
    print(f"Archive members: {stats['archive_members']:,}")
    print(f"Skipped from prior run: {stats['members_skipped']:,}")
    print(f"Members with facts: {stats['members_written']:,}")
    print(f"Members without facts: {stats['members_empty']:,}")
    print(f"Failed members: {stats['members_failed']:,}")
    print(f"New fact rows written: {stats['fact_rows_written']:,}")
    print(f"New Parquet parts written: {stats['parts_written']:,}")
    print(f"Dataset directory: {args.output_dir}")
    print(f"Manifest: {args.manifest}")

    if args.failures.exists():
        print(f"Failure log: {args.failures}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
