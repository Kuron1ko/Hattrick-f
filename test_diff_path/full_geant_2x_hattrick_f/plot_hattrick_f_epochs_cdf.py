from __future__ import annotations

"""Compatibility entry point for the registry-backed CDF plotter.

Preferred syntax:
    --method Hattrick Hattrick-f --epoch 22 9

The earlier shorthand remains supported:
    --epochs 9 40
"""

from plot_registered_cdf import main


if __name__ == "__main__":
    main()
