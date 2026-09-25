from __future__ import annotations

import plot_final_universal_pareto_cdf as plot


plot.REPORT = plot.HERE / "final_rqp_slack_weight.json"
plot.OUTPUT_STEM = "final_rqp_slack_weight_cdf"
plot.SUBTITLE = (
    "ESM High-slack adaptive RQP versus Hattrick · "
    "one frozen configuration · Level-4 snapshots 400–499"
)
plot.SERIES = (
    ("Hattrick", "baseline"),
    ("RQP-S · strict ESM", "candidate"),
)
plot.COLORS = {
    "Hattrick": "#2F78D4",
    "RQP-S · strict ESM": "#F28E2B",
}
plot.LINESTYLES = {
    "Hattrick": "-",
    "RQP-S · strict ESM": "--",
}


if __name__ == "__main__":
    plot.main()
