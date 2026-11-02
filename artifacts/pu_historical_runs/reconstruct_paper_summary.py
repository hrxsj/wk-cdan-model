"""Rebuild the manuscript's PU WK-CDAN summary from archived run CSVs."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUTS = (
    ("raw_results_multiseed_200epoch.csv", False),
    ("raw_results_oc2_source_500epoch.csv", True),
    ("raw_results_hard_oc2_500epoch.csv", True),
)
KEY = ("seed", "source_condition", "target_condition")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    selected = {}
    for filename, replace_oc2 in INPUTS:
        for row in read_rows(HERE / filename):
            replaceable = (
                row["source_name"] != row["target_name"]
                and (row["source_name"] == "OC2" or row["target_name"] == "OC2")
            )
            if replace_oc2 != replaceable:
                continue
            selected[tuple(row[field] for field in KEY)] = row

    rows = [selected[key] for key in sorted(selected)]
    if len(rows) != 45:
        raise ValueError(f"Expected 45 unique task/seed rows, found {len(rows)}")

    by_seed = defaultdict(list)
    for row in rows:
        if row["source_name"] != row["target_name"]:
            by_seed[row["seed"]].append(float(row["accuracy"]))
    seed_means = [statistics.mean(by_seed[seed]) for seed in sorted(by_seed)]
    summary = [{
        "method": "WK-CDAN (archived run summary)",
        "cross_mean": statistics.mean(seed_means),
        "cross_std_sample": statistics.stdev(seed_means),
        "seeds": len(seed_means),
        "cross_tasks_per_seed": 6,
        "rows": len(rows),
    }]

    write_rows(HERE / "reconstructed_paper_rows.csv", rows)
    write_rows(HERE / "reconstructed_summary.csv", summary)
    print(f"Cross-condition mean +/- sample SD: {summary[0]['cross_mean']:.4f} +/- "
          f"{summary[0]['cross_std_sample']:.4f}%")


if __name__ == "__main__":
    main()
