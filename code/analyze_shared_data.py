"""Reconstruct the manuscript's accuracy, timing and arch-response tables."""
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GAUGES = ("sidewall_shortening", "crown_settlement", "invert_heave",
          "vertical_shortening", "upper_chord", "middle_chord", "lower_chord")


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_total(rows, gauge, mechanism="total", region="all"):
    return sum(float(row["net_mm"]) for row in rows
               if row["gauge"] == gauge and row["mechanism"] == mechanism
               and row["region"] == region)


def summarize(output):
    output.mkdir(parents=True, exist_ok=True)
    reference = read_csv(ROOT / "data/accuracy_reference/responses.csv")[-1]
    reference_sources = read_csv(ROOT / "data/accuracy_reference/sources.csv")
    comparisons = []
    for timing in read_csv(ROOT / "data/timings.csv"):
        folder = ROOT / "data" / timing["case"]
        response = read_csv(folder / "responses.csv")[-1]
        row = dict(case=timing["case"], coordinates=int(timing["coordinates"]),
                   displacement_error_percent=max(abs(100 * (float(response[g + "_mm"])
                       / float(reference[g + "_mm"]) - 1)) for g in GAUGES),
                   plastic_displacement_error_percent="", total_seconds="",
                   upper_chord_weak_plane_mm="")
        if (folder / "sources.csv").exists():
            sources = read_csv(folder / "sources.csv")
            row["plastic_displacement_error_percent"] = max(abs(100 * (
                source_total(sources, g) / source_total(reference_sources, g) - 1)) for g in GAUGES)
            row["upper_chord_weak_plane_mm"] = source_total(sources, "upper_chord", "joint")
            row["total_seconds"] = float(timing["solution_seconds"]) + float(timing["source_seconds"])
        comparisons.append(row)
    write_csv(output / "feature_comparison.csv", comparisons)

    records = read_csv(ROOT / "inputs/arch_observations.csv")
    observed = []
    for name, subtract_first in (("Cumulative readings", False), ("Since first reading", True)):
        values = {}
        for gauge in ("upper_chord", "crown_settlement"):
            readings = [float(row["deformation_mm"]) for row in records if row["gauge"] == gauge]
            values[gauge] = readings[-1] - (readings[0] if subtract_first else 0)
        observed.append(dict(mapping=name, crown_settlement_mm=values["crown_settlement"],
                             upper_chord_mm=values["upper_chord"],
                             crown_upper_ratio=values["crown_settlement"] / values["upper_chord"],
                             equal_radial_motion_ratio=""))
    predictions = [{key: row[key] for key in observed[0]}
                   for row in read_csv(ROOT / "data/arch_predictions.csv")]
    write_csv(output / "arch_comparison.csv", predictions + observed)

    resolution = []
    for case in ("network_coarse", "full_network", "network_steps24"):
        sources = read_csv(ROOT / "data" / case / "sources.csv")
        for gauge in ("crown_settlement", "invert_heave"):
            resolution.append(dict(case=case, gauge=gauge,
                beyond_5m_percent=100 * source_total(sources, gauge, region="depth_over_5m")
                / source_total(sources, gauge)))
    write_csv(output / "depth_comparison.csv", resolution)
    return comparisons, observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons, _ = summarize(args.output)
    for row in comparisons:
        print(f"{row['case']}: displacement error {row['displacement_error_percent']:.3f}%")


if __name__ == "__main__":
    main()
