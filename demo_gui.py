"""Drive the real GUI widgets and print what the window displays.

The synthetic-click screenshot route is unreliable with Tk, so this exercises
the same widgets and handlers directly and reports the values the window would
show. It is a demonstration aid, not part of the test suite.
"""

from __future__ import annotations

import tkinter as tk

import black_scholes.gui as gui_module
from black_scholes.gui import BlackScholesGUI


def show(app: BlackScholesGUI, root: tk.Tk, label: str) -> None:
    root.update()
    print(f"--- {label} ---")
    print(f"  Option Price  : {app.price_var.get()}")
    print(f"  Delta         : {app.greek_vars['delta'].get()}")
    print(f"  Gamma         : {app.greek_vars['gamma'].get()}")
    print(f"  Vega (per 1%) : {app.greek_vars['vega'].get()}")
    print(f"  Theta (per d) : {app.greek_vars['theta'].get()}")
    print(f"  Rho   (per 1%): {app.greek_vars['rho'].get()}")
    print(f"  Monte Carlo   : {app.mc_var_label.get() or '(not run)'}")
    print(f"  Status bar    : {app.status_var.get()}")
    print()


def set_field(app: BlackScholesGUI, key: str, value: str) -> None:
    app.entries[key].delete(0, tk.END)
    app.entries[key].insert(0, value)


def main() -> None:
    # Never let a modal dialog block the demonstration.
    gui_module.messagebox.showerror = lambda *a, **k: None
    gui_module.messagebox.showinfo = lambda *a, **k: None

    root = tk.Tk()
    app = BlackScholesGUI(root)
    root.update()

    app._on_price()
    show(app, root, "CALL  S=100 K=100 T=1 r=5% vol=20%  (defaults)")

    app.option_type_var.set("put")
    app._on_price()
    show(app, root, "PUT   same inputs")

    app.option_type_var.set("call")
    app.mc_var.set(True)
    app._on_price()
    show(app, root, "CALL with 100k-path Monte Carlo validation")

    app.mc_var.set(False)
    set_field(app, "spot", "120")
    set_field(app, "strike", "110")
    set_field(app, "vol", "0.35")
    set_field(app, "rate", "0.06")
    app._on_price()
    show(app, root, "CALL  S=120 K=110 vol=35% r=6%")

    set_field(app, "spot", "-50")
    app._on_price()
    show(app, root, "INVALID INPUT (spot=-50)  -> must not crash or alter the display")

    app._on_clear()
    show(app, root, "AFTER CLEAR")

    root.destroy()
    print("GUI drive-through completed: no crashes, no blocking dialogs.")


if __name__ == "__main__":
    main()
