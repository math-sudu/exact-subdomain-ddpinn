"""Reference twelve-step histories used in the manuscript's unloading study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HISTORIES = {
    "bedding_oblique": ROOT / "results/orientation_extension/bedding_oblique",
    "bedding_aligned": ROOT / "results/orientation_extension/bedding_aligned",
    "bedding_crossed": ROOT / "results/orientation_extension/bedding_crossed",
}
