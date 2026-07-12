"""Extract pilot-company facts from the SEC bulk Company Facts ZIP.

The script reads ``companyfacts.zip`` directly, keeps only CIKs listed in the
pilot universe, flattens each SEC JSON member, and writes one processed CSV per
company. Existing processed files are skipped unless ``--overwrite`` is used.

Example
-------
PowerShell:
    python src/extract_pilot_companyfacts.py

    python src/extract_pilot_companyfacts.py `
        --pilot data/processed/universe/pilot_universe.csv `
        --archive data/raw/sec_bulk/companyfacts.zip
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

from fetch_companyfacts import (
    filter_forms,
    flatten_companyfacts,
)
from fetch_submissions import normalize_cik


CIK_PATTERN = re.compile(r"CIK(\d{10})", flags=re.IGNORECASE)

DEFAULT_PILOT = Path(
    "data/processed/universe/pilot_universe.csv"
)
DEFAULT_ARCHIVE = Path(
    "data/raw/sec_bulk/companyfacts.zip"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/processed/companyfacts"
)
DEFAULT_MISSING_OUTPUT = Path(
    "data/processed/companyfacts/"
    "pilot_companyfacts_missing.csv"
)
DEFAULT_FAILURE_OUTPUT = Path(
    "data/processed/companyfacts/"
    "pilot_companyfacts_failures.csv"
)


def read_pilot_ciks(path: Path) -> pd.DataFrame:
    """Read the pilot universe and return unique normalized CIKs."""
    pilot = pd.read_csv(
        path,
        dtype={"cik": str},
    )

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


def build_member_index(
    archive: zipfile.ZipFile,
) -> dict[str, zipfile.ZipInfo]:
    """Map CIKs to Company Facts JSON members."""
    result: dict[str, zipfile.ZipInfo] = {}

    for member in archive.infolist():
        if member.is_dir():
            continue

        if not member.filename.lower().endswith(".json"):
            continue

        cik = cik_from_member_name(member.filename)

        if cik:
            result[cik] = member

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Company Facts for pilot CIKs from the SEC "
            "bulk archive."
        )
    )
    parser.add_argument(
        "--pilot",
        type=Path,
        default=DEFAULT_PILOT,
        help=f"Pilot universe CSV. Default: {DEFAULT_PILOT}",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"SEC Company Facts ZIP. Default: {DEFAULT_ARCHIVE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for one flattened CSV per company. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
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
        "--overwrite",
        action="store_true",
        help="Replace processed files that already exist.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help=(
            "Print progress after this many pilot companies. "
            "Use 0 to disable."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        pilot = read_pilot_ciks(args.pilot)
        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        taxonomy_filter = args.taxonomy or None

        extracted = 0
        skipped = 0
        missing: list[dict[str, str]] = []
        failures: list[dict[str, str]] = []
        observations = 0

        with zipfile.ZipFile(args.archive) as archive:
            member_index = build_member_index(archive)

            for index, row in pilot.iterrows():
                cik = row["cik"]
                company = str(row["company"])
                output_path = (
                    args.output_dir
                    / f"CIK{cik}_companyfacts.csv"
                )

                if output_path.exists() and not args.overwrite:
                    skipped += 1
                else:
                    member = member_index.get(cik)

                    if member is None:
                        missing.append(
                            {
                                "cik": cik,
                                "company": company,
                                "reason": (
                                    "CIK not present in "
                                    "companyfacts.zip"
                                ),
                            }
                        )
                    else:
                        try:
                            with archive.open(member) as source:
                                payload = json.load(source)

                            facts = flatten_companyfacts(
                                payload,
                                taxonomy_filter=taxonomy_filter,
                            )
                            facts = filter_forms(
                                facts,
                                args.forms,
                            )

                            facts.to_csv(
                                output_path,
                                index=False,
                            )

                            extracted += 1
                            observations += len(facts)

                        except (
                            json.JSONDecodeError,
                            KeyError,
                            TypeError,
                            ValueError,
                        ) as exc:
                            failures.append(
                                {
                                    "cik": cik,
                                    "company": company,
                                    "error": str(exc),
                                }
                            )

                completed = index + 1

                if (
                    args.progress_every > 0
                    and completed % args.progress_every == 0
                ):
                    print(
                        f"Processed {completed:,}/{len(pilot):,} "
                        f"pilot companies"
                    )

        if missing:
            args.missing_output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            pd.DataFrame(missing).to_csv(
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

    print()
    print(f"Pilot companies: {len(pilot):,}")
    print(f"Extracted: {extracted:,}")
    print(f"Skipped existing: {skipped:,}")
    print(f"Missing from archive: {len(missing):,}")
    print(f"Failed: {len(failures):,}")
    print(f"Fact observations written: {observations:,}")
    print(f"Output directory: {args.output_dir}")

    if missing:
        print(f"Missing-CIK log: {args.missing_output}")

    if failures:
        print(f"Failure log: {args.failure_output}")

    return 0 if extracted + skipped > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
