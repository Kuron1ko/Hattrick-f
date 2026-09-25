from __future__ import annotations

import plot_final_universal_pareto_cdf as plot


plot.REPORT = plot.HERE / "final_robust_priority.json"
plot.OUTPUT_STEM = "final_robust_priority_cdf"
plot.SUBTITLE = (
    "Residual-quantile robust priority routing versus Hattrick · "
    "one frozen configuration · Level-4 snapshots 400–499"
)
plot.SERIES = (
    ("Hattrick", "baseline"),
    ("RQP · strict ESM", "candidate"),
)
plot.COLORS = {
    "Hattrick": "#2F78D4",
    "RQP · strict ESM": "#F28E2B",
}
plot.LINESTYLES = {
    "Hattrick": "-",
    "RQP · strict ESM": "--",
}


if __name__ == "__main__":
    plot.main()
