"""Create a reproducible pilot universe for the SEC distress model.

The pilot contains:
- every eligible filer with an Item 1.03 distress disclosure;
- a fixed-size random sample of eligible non-distress filers.

Eligibility:
- entityType == "operating";
- valid SIC code;
- SIC outside 6000-6999;
- not described as Asset-Backed Securities.

Example
-------
PowerShell:
    python src/build_pilot_universe.py

    python src/build_pilot_universe.py `
        --non-distress 2000 `
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "data/processed/universe/model_candidates.csv"
)
DEFAULT_OUTPUT = Path(
    "data/processed/universe/pilot_universe.csv"
)


def read_candidates(path: Path) -> pd.DataFrame:
    """Read and validate the candidate universe."""
    candidates = pd.read_csv(
        path,
        dtype={
            "cik": str,
            "sic": str,
        },
    )

    required = {
        "cik",
        "company",
        "entityType",
        "sic",
        "sicDescription",
        "hasDistressEvent",
    }
    missing = required.difference(candidates.columns)

    if missing:
        raise ValueError(
            f"Input file is missing columns: {sorted(missing)}"
        )

    candidates["cik"] = candidates["cik"].str.zfill(10)
    candidates["hasDistressEvent"] = pd.to_numeric(
        candidates["hasDistressEvent"],
        errors="raise",
    ).astype(int)

    return candidates


def eligible_universe(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the version-one corporate modelling scope."""
    result = candidates.copy()

    sic_numeric = pd.to_numeric(
        result["sic"],
        errors="coerce",
    )

    is_operating = (
        result["entityType"]
        .fillna("")
        .str.strip()
        .str.casefold()
        .eq("operating")
    )

    is_financial = sic_numeric.between(
        6000,
        6999,
        inclusive="both",
    )

    is_asset_backed = (
        result["sicDescription"]
        .fillna("")
        .str.contains(
            "Asset-Backed Securities",
            case=False,
            regex=False,
        )
    )

    keep = (
        is_operating
        & sic_numeric.notna()
        & ~is_financial
        & ~is_asset_backed
    )

    result = result.loc[keep].copy()
    result["sicNumeric"] = sic_numeric.loc[keep].astype(int)

    return result


def build_pilot(
    eligible: pd.DataFrame,
    non_distress_count: int,
    seed: int,
) -> pd.DataFrame:
    """Keep all distress filers and sample non-distress filers."""
    if non_distress_count < 0:
        raise ValueError("--non-distress cannot be negative.")

    distress = eligible.loc[
        eligible["hasDistressEvent"].eq(1)
    ].copy()

    non_distress_pool = eligible.loc[
        eligible["hasDistressEvent"].eq(0)
    ].copy()

    sample_size = min(
        non_distress_count,
        len(non_distress_pool),
    )

    non_distress = non_distress_pool.sample(
        n=sample_size,
        random_state=seed,
        replace=False,
    ).copy()

    distress["pilotGroup"] = "distress"
    non_distress["pilotGroup"] = "sampled_non_distress"

    distress["filerSelectionProbability"] = 1.0

    non_distress_probability = (
        sample_size / len(non_distress_pool)
        if len(non_distress_pool) > 0
        else 0.0
    )
    non_distress["filerSelectionProbability"] = (
        non_distress_probability
    )

    pilot = pd.concat(
        [distress, non_distress],
        ignore_index=True,
    )

    pilot["samplingSeed"] = seed

    pilot = (
        pilot.sort_values(
            ["hasDistressEvent", "company", "cik"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )

    return pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a pilot universe with all eligible distress "
            "filers and sampled non-distress filers."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input candidate universe. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output pilot universe. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--non-distress",
        type=int,
        default=2000,
        help=(
            "Number of eligible non-distress filers to sample. "
            "Default: 2000."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random sampling seed. Default: 42.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        candidates = read_candidates(args.input)
        eligible = eligible_universe(candidates)
        pilot = build_pilot(
            eligible=eligible,
            non_distress_count=args.non_distress,
            seed=args.seed,
        )

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        pilot.to_csv(args.output, index=False)

    except (
        FileNotFoundError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    eligible_distress = eligible["hasDistressEvent"].eq(1).sum()
    eligible_non_distress = eligible["hasDistressEvent"].eq(0).sum()

    pilot_distress = pilot["hasDistressEvent"].eq(1).sum()
    pilot_non_distress = pilot["hasDistressEvent"].eq(0).sum()

    print(f"Eligible filers: {len(eligible):,}")
    print(f"  Distress filers: {eligible_distress:,}")
    print(f"  Non-distress filers: {eligible_non_distress:,}")
    print()
    print(f"Pilot filers: {len(pilot):,}")
    print(f"  Distress filers retained: {pilot_distress:,}")
    print(f"  Non-distress filers sampled: {pilot_non_distress:,}")
    print(f"  Random seed: {args.seed}")
    print(f"Saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
