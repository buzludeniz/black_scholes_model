"""Launch the Black-Scholes GUI in its own process (for manual inspection)."""

import sys

from black_scholes.gui import run_gui

if __name__ == "__main__":
    sys.exit(run_gui())
