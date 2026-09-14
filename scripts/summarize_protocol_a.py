import re
import math
import csv
from pathlib import Path
from statistics import mean, stdev


SEEDS = [111, 222, 333, 444, 555]

METRICS = [
    "Accuracy",
    "Balanced Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
]

# 95% CI, t distribution, df = 4
T_CRITICAL = 2.776


def extract_metric(text, metric_name):
    pattern = rf"^{re.escape(metric_name)}:\s*([0-9.]+)"
    match = re.search(pattern, text, re.MULTILINE)

    if not match:
        raise ValueError(f"Could not find: {metric_name}")

    return float(match.group(1))


def main():
    results = {metric: [] for metric in METRICS}

    print("Protocol A Results")
    print("-------------------------")

    for seed in SEEDS:
        file_path = Path(f"results/protocol_a_seed{seed}.txt")

        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

        text = file_path.read_text()

        print(f"\nSeed {seed}")

        for metric in METRICS:
            value = extract_metric(text, metric)
            results[metric].append(value)
            print(f"{metric}: {value:.4f}")

    print()
    print("Summary")
    print("-------------------------")

    summary_rows = []

    for metric in METRICS:
        values = results[metric]

        metric_mean = mean(values)
        metric_sd = stdev(values)

        margin = T_CRITICAL * metric_sd / math.sqrt(len(values))

        ci_lower = metric_mean - margin
        ci_upper = metric_mean + margin

        print(
            f"{metric}: "
            f"Mean={metric_mean:.4f}, "
            f"SD={metric_sd:.4f}, "
            f"95% CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
        )

        summary_rows.append([
            metric,
            *values,
            metric_mean,
            metric_sd,
            ci_lower,
            ci_upper
        ])

    output_path = Path("results/protocol_a_summary.csv")

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "metric",
            "seed111",
            "seed222",
            "seed333",
            "seed444",
            "seed555",
            "mean",
            "sd",
            "ci_lower",
            "ci_upper"
        ])

        writer.writerows(summary_rows)

    print()
    print(f"Summary saved to: {output_path}")


if __name__ == "__main__":
    main()