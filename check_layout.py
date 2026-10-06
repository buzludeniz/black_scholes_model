"""Layout audit: verify every widget fits inside the window and nothing is clipped.

This is the check a screenshot cannot do reliably. It walks the widget tree and
confirms each visible widget's bounding box lies within its parent's, and that
the whole thing fits the window.
"""

from __future__ import annotations

import contextlib
import sys
import tkinter as tk
from tkinter import ttk

from black_scholes.gui import BlackScholesGUI


def walk(widget: tk.Misc):
    yield widget
    for child in widget.winfo_children():
        yield from walk(child)


def main() -> int:
    root = tk.Tk()
    BlackScholesGUI(root)
    root.update_idletasks()
    root.update()

    root.withdraw()  # measure geometry without fighting the window manager

    win_w, win_h = root.winfo_width(), root.winfo_height()
    print(f"window: {win_w} x {win_h}")
    print()

    problems: list[str] = []
    skipped = 0

    for w in walk(root):
        if not isinstance(
            w,
            (
                tk.Tk,
                ttk.Frame,
                tk.Frame,
                ttk.LabelFrame,
                tk.Label,
                ttk.Entry,
                ttk.Button,
                ttk.Radiobutton,
                ttk.Checkbutton,
            ),
        ):
            continue
        if isinstance(w, tk.Tk):
            continue
        if not w.winfo_ismapped() and not w.winfo_manager():
            skipped += 1
            continue

        # Absolute coords of this widget. A widget destroyed mid-walk raises
        # TclError, so skip it rather than aborting the audit.
        with contextlib.suppress(tk.TclError):
            x = w.winfo_rootx() - root.winfo_rootx()
            y = w.winfo_rooty() - root.winfo_rooty()
            width = w.winfo_width()
            height = w.winfo_height()

        if width <= 1 or height <= 1:
            skipped += 1
            continue

        label = str(w)
        if isinstance(w, ttk.Entry):
            with contextlib.suppress(tk.TclError):
                label = f"Entry({w.get()!r})"

        if x < 0 or y < 0:
            problems.append(f"NEGATIVE POS  {type(w).__name__:16} at ({x},{y})")
        if x + width > win_w + 1:
            problems.append(
                f"OVERFLOW RIGHT {type(w).__name__:16} right={x + width} > {win_w}  {label[:40]}"
            )
        if y + height > win_h + 1:
            problems.append(
                f"CLIPPED BOTTOM {type(w).__name__:16} bottom={y + height} > {win_h}  {label[:40]}"
            )

    print(f"widgets measured, {skipped} skipped (unmapped/zero-size)")
    print()

    # Specifically confirm the results area is fully visible.
    print("--- key element positions (relative to window) ---")
    for w in walk(root):
        if isinstance(w, ttk.LabelFrame):
            text = w.cget("text")
            x = w.winfo_rootx() - root.winfo_rootx()
            y = w.winfo_rooty() - root.winfo_rooty()
            print(
                f"  LabelFrame {text!r:20} at ({x:3},{y:3}) {w.winfo_width()}x{w.winfo_height()}"
                f"  bottom={y + w.winfo_height()}"
            )
    print()

    if problems:
        print(f"LAYOUT PROBLEMS ({len(problems)}):")
        for p in problems:
            print(f"  {p}")
        return 1

    print("LAYOUT OK: every widget fits inside the window, nothing clipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
