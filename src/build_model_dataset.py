"""Build a small annual corporate-distress modelling dataset.

The script matches SEC Company Facts observations to the exact 10-K accession
that disclosed them, creates a restrained set of financial features, and labels
whether an Item 1.03 distress disclosure occurred within the next 12 months.

Example
-------
python src/build_model_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fetch_submissions import normalize_cik


# Candidate concepts are ordered from preferred to fallback.
INSTANT_FEATURES: dict[str, list[str]] = {
    "assets": [
        "Assets",
    ],
    "liabilities": [
        "Liabilities",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
}

DURATION_FEATURES: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
    "interest_expense": [
        "InterestExpenseNonOperating",
        "InterestExpense",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],
}

RAW_FEATURE_NAMES = list(INSTANT_FEATURES) + list(DURATION_FEATURES)
ANNUAL_MIN_DAYS = 250
ANNUAL_MAX_DAYS = 450
TARGET_HORIZON_DAYS = 365


def read_csv_with_cik(path: Path) -> pd.DataFrame:
    """Read a CSV while preserving CIKs as strings."""
    return pd.read_csv(
        path,
        dtype={
            "cik": str,
            "accessionNumber": str,
        },
    )


def normalize_cik_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the CIK column without mutating the input."""
    result = frame.copy()
    result["cik"] = result["cik"].map(normalize_cik)
    return result


def read_companyfacts_directory(directory: Path) -> pd.DataFrame:
    """Read and combine all processed company-facts CSV files."""
    paths = sorted(directory.glob("*_companyfacts.csv"))

    if not paths:
        raise FileNotFoundError(
            f"No '*_companyfacts.csv' files found in {directory}."
        )

    frames: list[pd.DataFrame] = []

    for path in paths:
        facts = read_csv_with_cik(path)
        facts["sourceFile"] = path.name
        frames.append(facts)

    combined = pd.concat(frames, ignore_index=True)
    combined = normalize_cik_column(combined)

    for column in ("start", "end", "filed"):
        combined[column] = pd.to_datetime(
            combined[column],
            errors="coerce",
        )

    combined["value"] = pd.to_numeric(
        combined["value"],
        errors="coerce",
    )

    return combined


def prepare_annual_filings(filings: pd.DataFrame) -> pd.DataFrame:
    """Select original 10-K filings and create one row per accession."""
    required = {
        "cik",
        "companyName",
        "filingDate",
        "reportDate",
        "form",
        "accessionNumber",
    }
    missing = required.difference(filings.columns)

    if missing:
        raise ValueError(
            f"Filings file is missing columns: {sorted(missing)}"
        )

    result = normalize_cik_column(filings)

    for column in ("filingDate", "reportDate"):
        result[column] = pd.to_datetime(
            result[column],
            errors="coerce",
        )

    # Use the original filing date as the point-in-time information date.
    # Amendments can be handled explicitly in a later version.
    result = result.loc[
        result["form"].fillna("").str.upper().eq("10-K")
    ].copy()

    result = result.dropna(
        subset=["cik", "accessionNumber", "filingDate", "reportDate"]
    )

    result = (
        result.sort_values(["cik", "filingDate", "accessionNumber"])
        .drop_duplicates(["cik", "accessionNumber"], keep="first")
        .reset_index(drop=True)
    )

    return result


def _best_concept_observation(
    filing_facts: pd.DataFrame,
    candidate_concepts: Iterable[str],
    report_date: pd.Timestamp,
    is_duration: bool,
) -> tuple[float, str | None]:
    """Return the preferred usable value and the concept that supplied it."""
    for priority, concept in enumerate(candidate_concepts):
        candidates = filing_facts.loc[
            filing_facts["concept"].eq(concept)
            & filing_facts["unit"].eq("USD")
            & filing_facts["end"].eq(report_date)
            & filing_facts["value"].notna()
        ].copy()

        if candidates.empty:
            continue

        candidates["conceptPriority"] = priority

        if is_duration:
            candidates = candidates.loc[
                candidates["start"].notna()
            ].copy()

            candidates["durationDays"] = (
                candidates["end"] - candidates["start"]
            ).dt.days

            candidates = candidates.loc[
                candidates["durationDays"].between(
                    ANNUAL_MIN_DAYS,
                    ANNUAL_MAX_DAYS,
                    inclusive="both",
                )
            ].copy()

            if candidates.empty:
                continue

            candidates["periodDistance"] = (
                candidates["durationDays"] - TARGET_HORIZON_DAYS
            ).abs()
        else:
            # Instant facts normally have no start date. Prefer those records,
            # but tolerate populated start dates if the API supplies them.
            candidates["periodDistance"] = (
                candidates["start"].notna().astype(int)
            )

        candidates["hasFramePenalty"] = (
            candidates["frame"].isna().astype(int)
            if "frame" in candidates.columns
            else 1
        )

        candidates = candidates.sort_values(
            [
                "conceptPriority",
                "periodDistance",
                "hasFramePenalty",
                "filed",
            ],
            ascending=[True, True, True, False],
        )

        chosen = candidates.iloc[0]
        return float(chosen["value"]), concept

    return np.nan, None


def build_annual_feature_table(
    annual_filings: pd.DataFrame,
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """Match facts to exact filing accessions and construct raw features."""
    facts_by_accession = {
        key: group
        for key, group in facts.groupby(
            ["cik", "accessionNumber"],
            sort=False,
        )
    }

    rows: list[dict[str, object]] = []

    for filing in annual_filings.itertuples(index=False):
        key = (filing.cik, filing.accessionNumber)
        filing_facts = facts_by_accession.get(key)

        row: dict[str, object] = {
            "cik": filing.cik,
            "companyName": filing.companyName,
            "filingDate": filing.filingDate,
            "reportDate": filing.reportDate,
            "accessionNumber": filing.accessionNumber,
        }

        if filing_facts is None:
            for feature_name in RAW_FEATURE_NAMES:
                row[feature_name] = np.nan
                row[f"{feature_name}_concept"] = None
        else:
            for feature_name, concepts in INSTANT_FEATURES.items():
                value, source_concept = _best_concept_observation(
                    filing_facts=filing_facts,
                    candidate_concepts=concepts,
                    report_date=filing.reportDate,
                    is_duration=False,
                )
                row[feature_name] = value
                row[f"{feature_name}_concept"] = source_concept

            for feature_name, concepts in DURATION_FEATURES.items():
                value, source_concept = _best_concept_observation(
                    filing_facts=filing_facts,
                    candidate_concepts=concepts,
                    report_date=filing.reportDate,
                    is_duration=True,
                )
                row[feature_name] = value
                row[f"{feature_name}_concept"] = source_concept

        rows.append(row)

    features = pd.DataFrame(rows)
    return add_financial_ratios(features)


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    positive_denominator: bool = False,
) -> pd.Series:
    """Divide while returning missing values for unusable denominators."""
    valid = denominator.notna() & numerator.notna()

    if positive_denominator:
        valid &= denominator.gt(0)
    else:
        valid &= denominator.ne(0)

    result = pd.Series(
        np.nan,
        index=numerator.index,
        dtype=float,
    )
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def add_financial_ratios(features: pd.DataFrame) -> pd.DataFrame:
    """Create a small, interpretable first set of financial ratios."""
    result = features.copy()

    result["liabilities_to_assets"] = safe_divide(
        result["liabilities"],
        result["assets"],
        positive_denominator=True,
    )
    result["equity_to_assets"] = safe_divide(
        result["equity"],
        result["assets"],
        positive_denominator=True,
    )
    result["current_ratio"] = safe_divide(
        result["current_assets"],
        result["current_liabilities"],
        positive_denominator=True,
    )
    result["cash_to_assets"] = safe_divide(
        result["cash"],
        result["assets"],
        positive_denominator=True,
    )
    result["return_on_assets"] = safe_divide(
        result["net_income"],
        result["assets"],
        positive_denominator=True,
    )
    result["operating_margin"] = safe_divide(
        result["operating_income"],
        result["revenue"],
        positive_denominator=True,
    )
    result["operating_cash_flow_to_assets"] = safe_divide(
        result["operating_cash_flow"],
        result["assets"],
        positive_denominator=True,
    )
    result["interest_coverage"] = safe_divide(
        result["operating_income"],
        result["interest_expense"].abs(),
        positive_denominator=True,
    )
    result["log_assets"] = np.where(
        result["assets"].gt(0),
        np.log(result["assets"]),
        np.nan,
    )

    result["raw_features_available"] = result[
        RAW_FEATURE_NAMES
    ].notna().sum(axis=1)

    return result


def add_distress_target(
    features: pd.DataFrame,
    events: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Label distress in (filing date, filing date + 365 days]."""
    result = features.copy()
    events = normalize_cik_column(events)

    events["filingDate"] = pd.to_datetime(
        events["filingDate"],
        errors="coerce",
    )
    event_dates = {
        cik: group["filingDate"].dropna().sort_values().to_numpy()
        for cik, group in events.groupby("cik")
    }

    targets: list[object] = []
    days_to_events: list[float] = []
    target_observed: list[bool] = []

    for row in result.itertuples(index=False):
        horizon_end = row.filingDate + pd.Timedelta(
            days=TARGET_HORIZON_DAYS
        )
        is_observed = horizon_end <= as_of_date
        target_observed.append(is_observed)

        dates = event_dates.get(row.cik, np.array([], dtype="datetime64[ns]"))
        later_events = dates[
            (dates > np.datetime64(row.filingDate))
            & (dates <= np.datetime64(horizon_end))
        ]

        if len(later_events) > 0:
            first_event = pd.Timestamp(later_events[0])
            targets.append(1)
            days_to_events.append(
                float((first_event - row.filingDate).days)
            )
        elif is_observed:
            targets.append(0)
            days_to_events.append(np.nan)
        else:
            targets.append(pd.NA)
            days_to_events.append(np.nan)

    result["targetObserved"] = target_observed
    result["distress12m"] = pd.array(targets, dtype="Int64")
    result["daysToDistress"] = days_to_events

    return result


def infer_as_of_date(
    filings: pd.DataFrame,
    events: pd.DataFrame,
    explicit_date: str | None,
) -> pd.Timestamp:
    """Determine the data cut-off date used for target observability."""
    if explicit_date:
        as_of_date = pd.Timestamp(explicit_date)
        if pd.isna(as_of_date):
            raise ValueError(f"Invalid --as-of-date: {explicit_date}")
        return as_of_date.normalize()

    date_candidates: list[pd.Timestamp] = []

    for frame in (filings, events):
        if "filingDate" not in frame.columns:
            continue

        dates = pd.to_datetime(
            frame["filingDate"],
            errors="coerce",
        ).dropna()

        if not dates.empty:
            date_candidates.append(dates.max())

    if not date_candidates:
        raise ValueError(
            "Could not infer an as-of date from the input files."
        )

    return max(date_candidates).normalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build annual SEC financial features and a 12-month "
            "Item 1.03 distress target."
        )
    )
    parser.add_argument(
        "--filings",
        type=Path,
        default=Path("data/processed/filings/all_filings.csv"),
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("data/processed/events/distress_events.csv"),
    )
    parser.add_argument(
        "--facts-dir",
        type=Path,
        default=Path("data/processed/companyfacts"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/model/model_dataset.csv"),
    )
    parser.add_argument(
        "--as-of-date",
        help=(
            "Optional YYYY-MM-DD data cut-off. Otherwise inferred from "
            "the latest filing date in the inputs."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        filings = read_csv_with_cik(args.filings)
        events = read_csv_with_cik(args.events)
        facts = read_companyfacts_directory(args.facts_dir)

        annual_filings = prepare_annual_filings(filings)
        features = build_annual_feature_table(
            annual_filings=annual_filings,
            facts=facts,
        )

        as_of_date = infer_as_of_date(
            filings=filings,
            events=events,
            explicit_date=args.as_of_date,
        )
        dataset = add_distress_target(
            features=features,
            events=events,
            as_of_date=as_of_date,
        )

        dataset = dataset.sort_values(
            ["filingDate", "cik"]
        ).reset_index(drop=True)

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        dataset.to_csv(args.output, index=False)

    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    observed = dataset["distress12m"].notna()
    positives = dataset["distress12m"].eq(1).sum()

    print(f"Annual 10-K rows: {len(dataset):,}")
    print(f"Rows with observed 12-month target: {observed.sum():,}")
    print(f"Positive distress targets: {positives:,}")
    print(f"Data as-of date: {as_of_date.date()}")
    print(f"Saved to: {args.output}")

    preview_columns = [
        "cik",
        "companyName",
        "filingDate",
        "assets",
        "liabilities_to_assets",
        "return_on_assets",
        "current_ratio",
        "distress12m",
        "targetObserved",
    ]
    print("\nPreview:")
    print(
        dataset[preview_columns]
        .tail(12)
        .to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
