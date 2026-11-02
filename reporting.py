"""Rebuild comparison tables and plots from experiment result files."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values):
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def summarize(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for path in sorted(source.rglob("raw_results.csv")):
        experiment = path.parent.relative_to(source).as_posix()
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                snr = row.get("snr_db", "")
                condition = "clean" if snr in {"", "None", "none", None} else f"{float(snr):g} dB"
                task = (row.get("source_hp") or row.get("source_name") or row.get("source_rpm") or row.get("source_condition"),
                        row.get("target_hp") or row.get("target_name") or row.get("target_rpm") or row.get("target_condition"))
                row["_task"] = "->".join(task)
                groups[(experiment, row["method"], condition)].append(row)
    if not groups:
        raise ValueError(f"No raw_results.csv records found under {source}")
    summaries, task_summaries = [], []
    for (experiment, method, condition), rows in sorted(groups.items()):
        observed = set()
        for row in rows:
            key = (row["seed"], row["_task"])
            if key in observed:
                raise ValueError(f"Duplicate experiment result: {experiment}, {method}, {key}")
            observed.add(key)
        by_repeat, by_task = defaultdict(list), defaultdict(list)
        for row in rows:
            by_repeat[row["seed"]].append(row)
            by_task[row["_task"]].append(row)
        task_sets = [{row["_task"] for row in repeat} for repeat in by_repeat.values()]
        if any(tasks != task_sets[0] for tasks in task_sets):
            raise ValueError(f"Initializations have incomplete or mismatched task sets: {experiment}, {method}")
        accuracy = stats([mean(float(row["accuracy"]) for row in repeat) for repeat in by_repeat.values()])
        f1 = stats([mean(float(row["macro_f1"]) for row in repeat) for repeat in by_repeat.values()])
        summaries.append({"experiment": experiment, "method": method, "condition": condition,
                          "accuracy_mean": accuracy[0], "accuracy_std": accuracy[1],
                          "macro_f1_mean": f1[0], "macro_f1_std": f1[1],
                          "repeats": len(by_repeat), "tasks_per_repeat": len(task_sets[0]), "runs": len(rows)})
        for task, records in sorted(by_task.items()):
            accuracy, f1 = stats([float(row["accuracy"]) for row in records]), stats([float(row["macro_f1"]) for row in records])
            task_summaries.append({"experiment": experiment, "method": method, "condition": condition,
                                   "task": task, "accuracy_mean": accuracy[0], "accuracy_std": accuracy[1],
                                   "macro_f1_mean": f1[0], "macro_f1_std": f1[1], "repeats": len(records)})
    write_csv(destination / "summary.csv", summaries)
    write_csv(destination / "task_summary.csv", task_summaries)
    (destination / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    def tex(value):
        return str(value).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    table = [r"\begin{tabular}{lllrr}", r"Experiment & Method & Condition & Accuracy (\%) & Macro-F1 (\%) \\", r"\hline"]
    for row in summaries:
        table.append(f"{tex(row['experiment'])} & {tex(row['method'])} & {tex(row['condition'])} & "
                     f"{row['accuracy_mean']:.2f} $\\pm$ {row['accuracy_std']:.2f} & "
                     f"{row['macro_f1_mean']:.2f} $\\pm$ {row['macro_f1_std']:.2f} " + r"\\")
    table.append(r"\end{tabular}")
    (destination / "comparison.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(max(7, len(summaries) * 0.7), 5))
    axis.bar(range(len(summaries)), [row["accuracy_mean"] for row in summaries],
             yerr=[row["accuracy_std"] for row in summaries], capsize=3, color="#337DA6")
    axis.set_xticks(range(len(summaries)))
    axis.set_xticklabels([f"{row['method']}\n{row['condition']}\n{row['experiment']}" for row in summaries],
                         rotation=35, ha="right", fontsize=8)
    axis.set_ylabel("Target accuracy (%)")
    axis.set_ylim(0, 105)
    figure.tight_layout()
    figure.savefig(destination / "accuracy.pdf")
    figure.savefig(destination / "accuracy.png", dpi=180)
    plt.close(figure)
    print(f"Exported {len(summaries)} method summaries and {len(task_summaries)} task summaries to {destination}")
    return summaries
