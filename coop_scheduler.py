#coop_scheduler.py
import argparse
import pandas as pd
from datetime import datetime, timedelta
import re
import random
import math

ELIGIBLE_KEYS = ["off", "career prep", "business management", "bus mgt", "management"]

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def is_eligible(subject: str) -> bool:
    s = normalize(subject)
    return any(k in s for k in ELIGIBLE_KEYS)

def parse_wide_schedule_str(schedule_str: str) -> dict:
    """
    Turns '1 - career prep, 2 - algebra, 3 - off, ...' into {1:'career prep', 2:'algebra', 3:'off', ...}
    """
    periods = {}
    if not isinstance(schedule_str, str) or not schedule_str.strip():
        return periods
    for entry in schedule_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # allow both "1 - X" and "1: X"
        m = re.match(r"^\s*(\d+)\s*[-:]\s*(.+?)\s*$", entry)
        if not m:
            continue
        p = int(m.group(1))
        subj = m.group(2).strip()
        periods[p] = subj
    return periods

def load_student_periods(df: pd.DataFrame) -> dict:
    """
    Supports TWO input shapes:
      A) 'wide' with columns: Name, Schedule   (Schedule is '1 - x, 2 - y, ...')
      B) 'long' with columns: Name, Period, Class
    Returns a dict:
      { name: {
          'eligible_periods': [ints],
          'off_periods': [ints],
          'career_prep_periods': [ints],
          'last_worked': None,
          'total_shifts': 0
        }, ... }
    """
    df_cols = [c.lower() for c in df.columns]

    # Try to detect shape B (long)
    long_ok = all(col in df_cols for col in ["name", "period", "class"])

    # Try to detect shape A (wide)
    wide_ok = all(col in df_cols for col in ["name", "schedule"])

    if not (long_ok or wide_ok):
        raise ValueError("Input must have either columns [Name, Schedule] or [Name, Period, Class].")

    student_periods = {}

    if wide_ok:
        # normalize columns
        df2 = df.rename(columns={c: c.lower() for c in df.columns})
        for _, row in df2.iterrows():
            name = str(row["name"]).strip()
            sched_map = parse_wide_schedule_str(row["schedule"])
            eligible = [p for p, subj in sched_map.items() if is_eligible(subj)]
            off_periods = [p for p, subj in sched_map.items() if "off" in normalize(subj)]
            career_prep_periods = [p for p, subj in sched_map.items() if "career prep" in normalize(subj)]
            student_periods[name] = {
                "eligible_periods": sorted(set(eligible)),
                "off_periods": sorted(set(off_periods)),
                "career_prep_periods": sorted(set(career_prep_periods)),
                "last_worked": None,
                "total_shifts": 0
            }

    if long_ok:
        df3 = df.rename(columns={c: c.lower() for c in df.columns})
        # Ensure Period is int
        df3["period"] = df3["period"].astype(int, errors="ignore")
        for name, grp in df3.groupby("name"):
            eligible, off_p, cp_p = [], [], []
            for _, r in grp.iterrows():
                p = int(r["period"])
                subj = str(r["class"])
                if is_eligible(subj):
                    eligible.append(p)
                nsubj = normalize(subj)
                if "off" in nsubj:
                    off_p.append(p)
                if "career prep" in nsubj:
                    cp_p.append(p)
            if name not in student_periods:
                student_periods[name] = {
                    "eligible_periods": [],
                    "off_periods": [],
                    "career_prep_periods": [],
                    "last_worked": None,
                    "total_shifts": 0
                }
            student_periods[name]["eligible_periods"] = sorted(set(student_periods[name]["eligible_periods"] + eligible))
            student_periods[name]["off_periods"] = sorted(set(student_periods[name]["off_periods"] + off_p))
            student_periods[name]["career_prep_periods"] = sorted(set(student_periods[name]["career_prep_periods"] + cp_p))

    return student_periods

def day_type_for(idx: int) -> str:
    # idx is the zero-based day counter in the generated schedule window
    return "A" if idx % 2 == 0 else "B"

def periods_for_daytype(day_type: str):
    return range(1, 5) if day_type == "A" else range(5, 9)

def next_school_day(d: datetime, skip_weekends: bool) -> datetime:
    if not skip_weekends:
        return d + timedelta(days=1)
    # move to next day; if Sat, jump to Mon; if Sun, jump to Mon
    nd = d + timedelta(days=1)
    while nd.weekday() >= 5:  # 5=Sat, 6=Sun
        nd += timedelta(days=1)
    return nd

def build_schedule(
    student_periods: dict,
    start_date: datetime,
    num_days: int,
    max_per_period: int = 2,
    min_gap_days: int = 2,
    skip_weekends: bool = True,
    seed: int = 42,
):
    rng = random.Random(seed)
    schedule_rows = []
    current_date = start_date
    days_scheduled = 0

    # One-per-day guard
    assigned_today = set()

    # Track per-period usage (optional but helps variety)
    per_period_counts = {name: {p: 0 for p in range(1, 9)} for name in student_periods}

    # Global fairness trackers
    n_students = max(1, len(student_periods))
    total_slots_planned = num_days * 4 * max_per_period  # 4 periods/day in your A/B layout
    assignments_done = 0  # increments every time we place one student in one period

    while days_scheduled < num_days:
        # Move to weekday if skipping weekends
        if skip_weekends and current_date.weekday() >= 5:
            current_date += timedelta(days=(7 - current_date.weekday()))

        assigned_today.clear()

        day_idx = days_scheduled
        day_type = day_type_for(day_idx)

        for period in periods_for_daytype(day_type):
            slots_filled_this_period = 0

            while slots_filled_this_period < max_per_period:
                # Build candidate set that passes hard requirements
                candidates = []
                for name, data in student_periods.items():
                    if period not in data["eligible_periods"]:
                        continue
                    if name in assigned_today:  # one shift per day
                        continue

                    last = data["last_worked"]
                    days_since = (current_date - last).days if last else float("inf")
                    if days_since < min_gap_days:
                        continue

                    total_so_far = data["total_shifts"]
                    # Fairness math: expected load at this moment
                    expected = assignments_done / n_students  # float
                    # If we assign this person now, their new total would be:
                    new_total = total_so_far + 1
                    # Hard cap: don't let anyone get > floor(expected)+1 ahead,
                    # unless we literally have no other options.
                    hard_cap_after = math.floor(expected) + 1
                    # We'll compute this as a filter later if we have alternatives.

                    # Preference signals (small weights; fairness drives 90% of choice)
                    off_bonus = 1 if period in data["off_periods"] else 0
                    per_period_penalty = per_period_counts[name][period]  # fewer is better

                    # Deficit = how far behind expectation this person is *now*
                    deficit = expected - total_so_far  # bigger -> more behind
                    # Secondary signal: longer since last worked
                    recency = days_since if days_since != float("inf") else 9999

                    candidates.append({
                        "name": name,
                        "deficit": deficit,
                        "off_bonus": off_bonus,
                        "total": total_so_far,
                        "new_total": new_total,
                        "hard_cap_after": hard_cap_after,
                        "recency": recency,
                        "per_period_penalty": per_period_penalty,
                    })

                if not candidates:
                    break  # no one available for this slot; move on

                # If we have at least one candidate who wouldn't exceed the cap, filter to those.
                non_capped = [c for c in candidates if c["new_total"] <= c["hard_cap_after"]]
                usable = non_capped if non_capped else candidates  # if all would exceed cap, allow the best of them

                # Sort by:
                # 1) larger deficit (more behind expected) first
                # 2) OFF bonus (prefer OFF a bit)
                # 3) fewer total shifts so far
                # 4) fewer times in this specific period
                # 5) longer since last worked
                # 6) small jitter
                rng.shuffle(usable)
                usable.sort(key=lambda c: (
                    -c["deficit"],
                    -c["off_bonus"],
                    c["total"],
                    c["per_period_penalty"],
                    -c["recency"],
                ))

                chosen = usable[0]

                # Place the assignment
                name = chosen["name"]
                schedule_rows.append({
                    "Date": current_date.strftime("%Y-%m-%d"),
                    "Day": day_type,
                    "Period": period,
                    "Student": name
                })

                # Update trackers
                student_periods[name]["last_worked"] = current_date
                student_periods[name]["total_shifts"] += 1
                per_period_counts[name][period] += 1
                assigned_today.add(name)

                assignments_done += 1
                slots_filled_this_period += 1

        days_scheduled += 1
        current_date = next_school_day(current_date, skip_weekends)

    return pd.DataFrame(schedule_rows)


def main():
    ap = argparse.ArgumentParser(description="CO-OP Shift Scheduler")
    ap.add_argument("--input", default="Employee Schedule.xlsx", help="Path to Excel with either [Name, Schedule] or [Name, Period, Class]")
    ap.add_argument("--out", default="final_shift_schedule.xlsx", help="Output Excel filename")
    ap.add_argument("--csv", default=None, help="Optional CSV output filename")
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=10, help="Number of school days to schedule")
    ap.add_argument("--max-per-period", type=int, default=2, help="Max students per period")
    ap.add_argument("--min-gap-days", type=int, default=2, help="Min days between shifts (OFF periods ignore this)")
    ap.add_argument("--skip-weekends", action="store_true", help="Skip Saturdays/Sundays")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for tie-breakers")
    args = ap.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")

    # Load Excel (first sheet)
    df = pd.read_excel(args.input)
    students = load_student_periods(df)

    schedule_df = build_schedule(
        students,
        start_date=start_date,
        num_days=args.days,
        max_per_period=args.max_per_period,
        min_gap_days=args.min_gap_days,
        skip_weekends=args.skip_weekends,
        seed=args.seed,
    )

    # Save
    with pd.ExcelWriter(args.out, engine="openpyxl") as writer:
        schedule_df.to_excel(writer, index=False, sheet_name="Schedule")
        # also include a small 'Summary' sheet with counts
        summary = schedule_df.groupby("Student").size().reset_index(name="Total Shifts").sort_values("Total Shifts", ascending=False)
        summary.to_excel(writer, index=False, sheet_name="Summary")

    if args.csv:
        schedule_df.to_csv(args.csv, index=False)

    print(f"Wrote schedule to {args.out}")
    if args.csv:
        print(f"Wrote CSV to {args.csv}")

if __name__ == "__main__":
    main()
