"""
Build a Celonis-like `_CEL_MERGED_ACTIVITIES` event log from multiple activity tables.

Implemented behavior (based on Celonis docs):
1) MERGE_EVENTLOG direction:
   - First input table is TARGET, second is SOURCE.
   - SOURCE activities are merged into TARGET cases (never the other way around).
2) Supports transitive mapping SOURCE_CASE_ID -> TARGET_CASE_ID.
3) MERGE_EVENTLOG_DISTINCT behavior:
   - Same as merge, but removes duplicated merged activities.
4) Activity sorting:
   - Primary: timestamp
   - Secondary: sorting column (if available)
   - If sorting is missing and timestamps are equal, order is unstable by nature;
     this script uses a stable fallback.
5) Day-based activities:
   - Activity type is considered day-based if all its timestamps are at 00:00:00.
   - For events on the same day in a case, day-based events are moved forward/downstream
     according to sorting values and then timestamp-adjusted to avoid negative durations.

Usage example:
  python scripts/build_cel_merged_activities.py \\
    --target-events data/raw/ACTIVITIES_TARGET.csv \\
    --target-case-col CASE_ID --target-activity-col ACTIVITY --target-ts-col TIMESTAMP \\
    --source-spec data/raw/source_specs.json \\
    --output data/translation/_CEL_MERGED_ACTIVITIES.csv \\
    --distinct

`source_specs.json` schema:
[
  {
    "name": "ACTIVITIES_BSEG",
    "events_path": "data/raw/ACTIVITIES_BSEG.csv",
    "case_id_col": "CASE_ID",
    "activity_col": "ACTIVITY",
    "timestamp_col": "TIMESTAMP",
    "sorting_col": "SORTING",
    "mapping_path": "data/raw/BSEG_to_BKPF_case_map.csv",
    "mapping_source_col": "SOURCE_CASE_ID",
    "mapping_target_col": "TARGET_CASE_ID"
  }
]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd


@dataclass
class EventTableSpec:
    name: str
    events_path: Path
    case_id_col: str
    activity_col: str
    timestamp_col: str
    sorting_col: Optional[str] = None
    mapping_path: Optional[Path] = None
    mapping_source_col: Optional[str] = None
    mapping_target_col: Optional[str] = None


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(path)
    raise ValueError(f"Unsupported file format for {path}")


def _normalize_events(
    df: pd.DataFrame,
    *,
    table_name: str,
    case_id_col: str,
    activity_col: str,
    timestamp_col: str,
    sorting_col: Optional[str],
) -> pd.DataFrame:
    cols = [case_id_col, activity_col, timestamp_col]
    if sorting_col:
        cols.append(sorting_col)

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name}: missing required columns {missing}")

    out = pd.DataFrame()
    out["CASE_ID"] = df[case_id_col].astype(str)
    out["ACTIVITY"] = df[activity_col].astype(str)
    out["TIMESTAMP"] = pd.to_datetime(df[timestamp_col], errors="coerce")
    out["SORTING"] = (
        pd.to_numeric(df[sorting_col], errors="coerce")
        if sorting_col and sorting_col in df.columns
        else pd.NA
    )
    out["SOURCE_TABLE"] = table_name
    out = out.dropna(subset=["CASE_ID", "ACTIVITY", "TIMESTAMP"]).copy()
    return out


def _map_source_to_target_cases(source_events: pd.DataFrame, spec: EventTableSpec) -> pd.DataFrame:
    if not spec.mapping_path:
        # If no mapping is provided, treat source cases as already target-aligned.
        out = source_events.copy()
        out["TARGET_CASE_ID"] = out["CASE_ID"]
        return out

    mapping = _read_table(spec.mapping_path)
    if not spec.mapping_source_col or not spec.mapping_target_col:
        raise ValueError(f"{spec.name}: mapping_source_col and mapping_target_col are required")

    for c in (spec.mapping_source_col, spec.mapping_target_col):
        if c not in mapping.columns:
            raise ValueError(f"{spec.name}: mapping column not found: {c}")

    mapping2 = mapping[[spec.mapping_source_col, spec.mapping_target_col]].copy()
    mapping2.columns = ["SOURCE_CASE_ID", "TARGET_CASE_ID"]
    mapping2["SOURCE_CASE_ID"] = mapping2["SOURCE_CASE_ID"].astype(str)
    mapping2["TARGET_CASE_ID"] = mapping2["TARGET_CASE_ID"].astype(str)

    out = source_events.merge(
        mapping2,
        left_on="CASE_ID",
        right_on="SOURCE_CASE_ID",
        how="inner",
    )
    out = out.drop(columns=["SOURCE_CASE_ID"])
    return out


def _is_midnight(ts: pd.Series) -> pd.Series:
    return (
        (ts.dt.hour == 0)
        & (ts.dt.minute == 0)
        & (ts.dt.second == 0)
        & (ts.dt.microsecond == 0)
    )


def _compute_day_based_activity_types(events: pd.DataFrame) -> Dict[str, bool]:
    # Day-based if an activity type always occurs at midnight in the full merged event log.
    by_activity = events.groupby("ACTIVITY", dropna=False)["TIMESTAMP"]
    return by_activity.apply(lambda s: bool(_is_midnight(s).all())).to_dict()


def _apply_activity_sorting_and_day_based_logic(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events

    out = events.copy()
    out["EVENT_DAY"] = out["TIMESTAMP"].dt.floor("D")
    out["SORTING_NORM"] = pd.to_numeric(out["SORTING"], errors="coerce")
    # Stable fallback for rows with null sorting (important for deterministic outputs).
    out["SORTING_NORM"] = out["SORTING_NORM"].fillna(10**12)
    out["ROW_ID"] = range(len(out))

    # Base ordering: timestamp, then sorting.
    out = out.sort_values(
        by=["CASE_ID", "EVENT_DAY", "TIMESTAMP", "SORTING_NORM", "ROW_ID"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    day_based_map = _compute_day_based_activity_types(out)

    adjusted_groups: List[pd.DataFrame] = []
    group_cols = ["CASE_ID", "EVENT_DAY"]
    for _, g in out.groupby(group_cols, sort=False, dropna=False):
        rows = g.to_dict("records")
        n = len(rows)

        # Reorder day-based activities within same day according to sorting value.
        i = 0
        while i < n - 1:
            curr = rows[i]
            is_day_based = day_based_map.get(str(curr["ACTIVITY"]), False)
            if not is_day_based:
                i += 1
                continue
            # Move down while current has larger sorting than next.
            j = i
            moved = False
            while j < n - 1 and rows[j]["SORTING_NORM"] > rows[j + 1]["SORTING_NORM"]:
                rows[j], rows[j + 1] = rows[j + 1], rows[j]
                j += 1
                moved = True

            # Timestamp adjustment rule for day-based event at new position:
            # assign timestamp from following activity on same day if available.
            if j < n - 1:
                if moved or not moved:
                    rows[j]["TIMESTAMP"] = rows[j + 1]["TIMESTAMP"]
            i += 1

        adjusted_groups.append(pd.DataFrame(rows))

    merged = pd.concat(adjusted_groups, ignore_index=True)
    merged = merged.sort_values(
        by=["CASE_ID", "TIMESTAMP", "SORTING_NORM", "ROW_ID"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return merged


def merge_eventlogs(
    target_spec: EventTableSpec,
    source_specs: Sequence[EventTableSpec],
    *,
    distinct: bool,
) -> pd.DataFrame:
    # Target anchors the output case space.
    target_raw = _read_table(target_spec.events_path)
    target_events = _normalize_events(
        target_raw,
        table_name=target_spec.name,
        case_id_col=target_spec.case_id_col,
        activity_col=target_spec.activity_col,
        timestamp_col=target_spec.timestamp_col,
        sorting_col=target_spec.sorting_col,
    )
    target_events["TARGET_CASE_ID"] = target_events["CASE_ID"]

    merged_parts: List[pd.DataFrame] = [target_events]

    for spec in source_specs:
        src_raw = _read_table(spec.events_path)
        src_events = _normalize_events(
            src_raw,
            table_name=spec.name,
            case_id_col=spec.case_id_col,
            activity_col=spec.activity_col,
            timestamp_col=spec.timestamp_col,
            sorting_col=spec.sorting_col,
        )
        mapped = _map_source_to_target_cases(src_events, spec)
        # Replace source case with target-oriented case (Celonis target-first merge semantics).
        mapped["CASE_ID"] = mapped["TARGET_CASE_ID"].astype(str)
        merged_parts.append(mapped[target_events.columns.tolist()])

    all_events = pd.concat(merged_parts, ignore_index=True)

    if distinct:
        all_events = all_events.drop_duplicates(
            subset=["CASE_ID", "ACTIVITY", "TIMESTAMP", "SORTING", "SOURCE_TABLE"]
        ).reset_index(drop=True)

    all_events = _apply_activity_sorting_and_day_based_logic(all_events)

    # Celonis-like convenience columns.
    all_events["_CASE_KEY"] = all_events["CASE_ID"].astype(str)
    all_events = all_events.rename(columns={"TIMESTAMP": "EVENTTIME"})
    all_events["_SORTING"] = all_events.groupby("CASE_ID").cumcount() + 1
    return all_events[
        [
            "_CASE_KEY",
            "CASE_ID",
            "ACTIVITY",
            "EVENTTIME",
            "_SORTING",
            "SORTING",
            "SOURCE_TABLE",
        ]
    ]


def _load_source_specs(path: Path) -> List[EventTableSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("source spec JSON must be a list")
    out: List[EventTableSpec] = []
    for row in data:
        out.append(
            EventTableSpec(
                name=str(row["name"]),
                events_path=Path(row["events_path"]),
                case_id_col=str(row["case_id_col"]),
                activity_col=str(row["activity_col"]),
                timestamp_col=str(row["timestamp_col"]),
                sorting_col=(str(row["sorting_col"]) if row.get("sorting_col") else None),
                mapping_path=(Path(row["mapping_path"]) if row.get("mapping_path") else None),
                mapping_source_col=(
                    str(row["mapping_source_col"]) if row.get("mapping_source_col") else None
                ),
                mapping_target_col=(
                    str(row["mapping_target_col"]) if row.get("mapping_target_col") else None
                ),
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Celonis-like _CEL_MERGED_ACTIVITIES from multiple activity tables."
    )
    parser.add_argument("--target-events", required=True, help="Path to target activity table")
    parser.add_argument("--target-name", default="TARGET_ACTIVITY_TABLE")
    parser.add_argument("--target-case-col", required=True)
    parser.add_argument("--target-activity-col", required=True)
    parser.add_argument("--target-ts-col", required=True)
    parser.add_argument("--target-sorting-col", default="")
    parser.add_argument(
        "--source-spec",
        required=True,
        help="JSON file describing source activity tables and mapping to target cases",
    )
    parser.add_argument("--output", required=True, help="Output path (.csv or .parquet)")
    parser.add_argument(
        "--distinct",
        action="store_true",
        help="Use MERGE_EVENTLOG_DISTINCT behavior (remove duplicates).",
    )
    args = parser.parse_args()

    target_spec = EventTableSpec(
        name=args.target_name,
        events_path=Path(args.target_events),
        case_id_col=args.target_case_col,
        activity_col=args.target_activity_col,
        timestamp_col=args.target_ts_col,
        sorting_col=args.target_sorting_col or None,
    )
    source_specs = _load_source_specs(Path(args.source_spec))

    merged = merge_eventlogs(target_spec, source_specs, distinct=args.distinct)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        merged.to_parquet(out_path, index=False)
    else:
        merged.to_csv(out_path, index=False)

    print(f"Wrote merged event log: {out_path}")
    print(f"Rows: {len(merged)}")


if __name__ == "__main__":
    main()

