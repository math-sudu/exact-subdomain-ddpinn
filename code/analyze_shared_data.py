"""Reconstruct the manuscript's accuracy, timing and arch-response tables."""
import argparse
import csv
from pathlib import Path
from statistics import mean


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


def arch_comparison(records, installation_states):
    """Pair cumulative day-50 readings with movements after three-quarters of release."""
    day50 = [row for row in records if int(row["day"]) == 50]
    measured = {gauge: mean(float(row["deformation_mm"]) for row in day50 if row["gauge"] == gauge)
                for gauge in ("crown_settlement", "upper_chord")}
    states = [row for row in installation_states if float(row["initial_release"]) == 0.75]
    design, = (row for row in states if row["mapping"] == "Equivalent-radius scale")
    nominal, = (row for row in states if row["mapping"] == "Clearance-width scale")
    pairs = [
        ("Day-50 cumulative readings", measured["crown_settlement"], measured["upper_chord"]),
        ("Support-design radius, full network", float(design["crown_settlement_mm"]),
         float(design["upper_chord_mm"])),
        ("Support-design radius, fine direct", float(design["crown_settlement_mm_direct"]),
         float(design["upper_chord_mm_direct"])),
        ("Nominal shaft width, full network", float(nominal["crown_settlement_mm"]),
         float(nominal["upper_chord_mm"])),
    ]
    return [dict(response=label, crown_settlement_mm=crown, upper_chord_mm=chord,
                 crown_upper_ratio=crown / chord) for label, crown, chord in pairs]


def forecast_comparison(reference):
    timings = {row["case"]: row for row in read_csv(ROOT / "data/cold_start_comparison.csv")}
    rows = []
    for case in ("early_forecast", "width_control", "current_enrichment", "cold_direct_coarse"):
        folder = ROOT / "data" / case
        row = {"case": case}
        for actual, target in zip(read_csv(folder / "responses.csv"), reference, strict=True):
            step = int(actual["step"])
            if step > 6:
                row[f"step_{step}_error_percent"] = max(abs(100 * (
                    float(actual[g + "_mm"]) / float(target[g + "_mm"]) - 1)) for g in GAUGES)
        row["upper_chord_slip_mm"] = source_total(read_csv(folder / "sources.csv"), "upper_chord", "joint")
        timing = timings[case]
        row.update(early_direct_seconds=timing["direct_prefix_s"], selection_seconds=timing["selection_s"],
                   fit_and_basis_seconds=(float(timing["feature_fit_s"]) + float(timing["basis_s"])
                                          if timing["feature_fit_s"] else ""),
                   late_equilibrium_seconds=timing["reduced_continuation_s"],
                   solver_seconds=timing["equilibrium_process_s"],
                   integration_seconds=timing["integration_process_s"], total_seconds=timing["whole_process_s"])
        rows.append(row)
    return rows


def material_comparison():
    cases = ("cold_direct_coarse", "ratio125", "ratio175", "cohesion008", "cohesion012",
             "friction22", "friction28", "pressure056", "pressure084")
    rows = []
    for case in cases:
        folder = ROOT / "data" / case
        final = read_csv(folder / "responses.csv")[-1]
        sources = read_csv(folder / "sources.csv")
        matrix_difference = (source_total(sources, "vertical_shortening", "matrix")
                             - source_total(sources, "sidewall_shortening", "matrix"))
        rows.append(dict(case=case, horizontal_mm=final["sidewall_shortening_mm"],
                         vertical_mm=final["vertical_shortening_mm"],
                         matrix_vertical_minus_horizontal_mm=matrix_difference,
                         upper_shear_mm=source_total(sources, "upper_chord", "joint")))
    return rows


def summarize(output):
    output.mkdir(parents=True, exist_ok=True)
    reference_history = read_csv(ROOT / "data/accuracy_reference/responses.csv")
    write_csv(output / "forecast_comparison.csv", forecast_comparison(reference_history))
    write_csv(output / "material_comparison.csv", material_comparison())
    reference = reference_history[-1]
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
    arch = arch_comparison(records, read_csv(ROOT / "data/arch_installation_states.csv"))
    write_csv(output / "arch_comparison.csv", arch)

    resolution = []
    for case in ("network_coarse", "full_network", "network_steps24"):
        sources = read_csv(ROOT / "data" / case / "sources.csv")
        for gauge in ("crown_settlement", "invert_heave"):
            resolution.append(dict(case=case, gauge=gauge,
                beyond_5m_percent=100 * source_total(sources, gauge, region="depth_over_5m")
                / source_total(sources, gauge)))
    write_csv(output / "depth_comparison.csv", resolution)
    return comparisons, arch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparisons, _ = summarize(args.output)
    for row in comparisons:
        print(f"{row['case']}: displacement error {row['displacement_error_percent']:.3f}%")


if __name__ == "__main__":
    main()
