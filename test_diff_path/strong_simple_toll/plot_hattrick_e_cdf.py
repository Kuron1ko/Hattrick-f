from __future__ import annotations

import plot_final_universal_pareto_cdf as plot


plot.REPORT = plot.HERE / "rqp_slack_esm_ablation.json"
plot.OUTPUT_STEM = "hattrick_e_cdf"
plot.TITLE = "CDF of NormFulFill"
plot.SUBTITLE = ""
plot.SERIES = (
    ("Hattrick", "baseline"),
    ("Hattrick-e", "raw_esm"),
)
plot.COLORS = {
    "Hattrick": "#2F78D4",
    "Hattrick-e": "#F28E2B",
}
plot.LINESTYLES = {
    "Hattrick": "-",
    "Hattrick-e": "--",
}
plot.LEGEND_NCOL = 2


if __name__ == "__main__":
    plot.main()
