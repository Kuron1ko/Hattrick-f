from __future__ import annotations

import plot_final_universal_pareto_cdf as plot


plot.REPORT = plot.HERE / "rqp_slack_kl_ablation.json"
plot.OUTPUT_STEM = "rqp_slack_kl_ablation_cdf"
plot.SUBTITLE = (
    "RQP-S strict one-variable ablation · raw ESM · with versus without "
    "the KL anchor · Level-4 snapshots 400–499"
)
plot.SERIES = (
    ("Hattrick", "baseline"),
    ("RQP-S · with KL", "with_kl"),
    ("RQP-S · without KL", "without_kl"),
)
plot.COLORS = {
    "Hattrick": "#2F78D4",
    "RQP-S · with KL": "#59A14F",
    "RQP-S · without KL": "#E15759",
}
plot.LINESTYLES = {
    "Hattrick": "-",
    "RQP-S · with KL": ":",
    "RQP-S · without KL": "--",
}
plot.LEGEND_NCOL = 3


if __name__ == "__main__":
    plot.main()
