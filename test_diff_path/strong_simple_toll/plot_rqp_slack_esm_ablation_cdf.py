from __future__ import annotations

import plot_final_universal_pareto_cdf as plot


plot.REPORT = plot.HERE / "rqp_slack_esm_ablation.json"
plot.OUTPUT_STEM = "rqp_slack_esm_ablation_cdf"
plot.SUBTITLE = (
    "RQP-S strict one-variable ablation · raw ESM versus residual-quantile "
    "correction · Level-4 snapshots 400–499"
)
plot.SERIES = (
    ("Hattrick", "baseline"),
    ("RQP-S · raw ESM", "raw_esm"),
    ("RQP-S · corrected ESM", "corrected_esm"),
)
plot.COLORS = {
    "Hattrick": "#2F78D4",
    "RQP-S · raw ESM": "#59A14F",
    "RQP-S · corrected ESM": "#F28E2B",
}
plot.LINESTYLES = {
    "Hattrick": "-",
    "RQP-S · raw ESM": ":",
    "RQP-S · corrected ESM": "--",
}
plot.LEGEND_NCOL = 3


if __name__ == "__main__":
    plot.main()
