"""
Simple Tkinter GUI for Black-Scholes Option Pricer.

A clean, functional desktop interface for pricing European options
and calculating implied volatility.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from black_scholes import (
    Greeks,
    OptionParams,
    OptionType,
    implied_volatility,
    monte_carlo_price,
    price_option,
)


class BlackScholesGUI:
    """Main GUI application."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Black-Scholes Option Pricer")
        self.root.geometry("560x680")
        self.root.minsize(560, 680)

        # Configure style
        self._setup_style()
        self._build_ui()

    def _setup_style(self) -> None:
        """Configure ttk styles for a clean look."""
        style = ttk.Style()
        style.theme_use("clam")

        # Colors
        bg = "#f5f5f5"
        fg = "#1a1a1a"
        accent = "#2563eb"
        accent_hover = "#1d4ed8"
        input_bg = "#ffffff"
        border = "#d1d5db"

        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 10))
        style.configure("TLabelFrame", background=bg, foreground=fg, font=("Segoe UI", 10, "bold"))
        style.configure("TLabelFrame.Label", background=bg, foreground=fg)
        style.configure(
            "TEntry",
            fieldbackground=input_bg,
            bordercolor=border,
            lightcolor=border,
            darkcolor=border,
        )
        style.configure("TCombobox", fieldbackground=input_bg, background=input_bg)
        style.configure("TRadiobutton", background=bg, foreground=fg, font=("Segoe UI", 10))
        style.configure("TCheckbutton", background=bg, foreground=fg, font=("Segoe UI", 10))

        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
            foreground="#ffffff",
            background=accent,
            borderwidth=0,
            focuscolor=accent,
        )
        style.map(
            "Accent.TButton",
            background=[("active", accent_hover), ("pressed", "#1e40af")],
            foreground=[("active", "#ffffff")],
        )

        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10),
            foreground="#374151",
            background="#e5e7eb",
            borderwidth=0,
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#d1d5db"), ("pressed", "#9ca3af")],
        )

        style.configure("Result.TLabel", font=("Consolas", 10), foreground="#111827")
        style.configure("ResultTitle.TLabel", font=("Segoe UI", 10, "bold"), foreground="#111827")

        self.root.configure(bg=bg)

    def _build_ui(self) -> None:
        """Build the complete UI."""
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main, text="Black-Scholes European Option Pricer", font=("Segoe UI", 14, "bold")
        )
        title.pack(pady=(0, 16))

        # Input section
        input_frame = ttk.LabelFrame(main, text="Parameters", padding=12)
        input_frame.pack(fill=tk.X, pady=(0, 12))

        self.entries = {}

        # Row 0: Option Type
        ttk.Label(input_frame, text="Option Type:", width=22, anchor=tk.W).grid(
            row=0, column=0, sticky=tk.W, pady=4, padx=(0, 8)
        )
        self.option_type_var = tk.StringVar(value="call")
        type_frame = ttk.Frame(input_frame)
        type_frame.grid(row=0, column=1, sticky=tk.W, pady=4)
        ttk.Radiobutton(type_frame, text="Call", variable=self.option_type_var, value="call").pack(
            side=tk.LEFT, padx=(0, 8)
        )
        ttk.Radiobutton(type_frame, text="Put", variable=self.option_type_var, value="put").pack(
            side=tk.LEFT
        )

        # Row 1-5: Numeric inputs
        fields = [
            ("Spot Price (S):", "spot", "100.0"),
            ("Strike Price (K):", "strike", "100.0"),
            ("Time to Maturity (years):", "time", "1.0"),
            ("Risk-Free Rate (r):", "rate", "0.05"),
            ("Volatility (σ):", "vol", "0.20"),
            ("Dividend Yield (q):", "div_yield", "0.0"),
        ]

        for i, (label, key, default) in enumerate(fields, start=1):
            ttk.Label(input_frame, text=label, width=22, anchor=tk.W).grid(
                row=i, column=0, sticky=tk.W, pady=4, padx=(0, 8)
            )
            entry = ttk.Entry(input_frame, width=18)
            entry.grid(row=i, column=1, sticky=tk.EW, pady=4)
            entry.insert(0, default)
            self.entries[key] = entry

        # Give the entry column the slack so labels keep a fixed width and the
        # inputs stretch with the window instead of the labels pushing them out.
        input_frame.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(8, 12))

        ttk.Button(
            btn_frame,
            text="Calculate Price & Greeks",
            style="Accent.TButton",
            command=self._on_price,
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(
            btn_frame, text="Implied Volatility", style="Secondary.TButton", command=self._on_iv
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="Clear", style="Secondary.TButton", command=self._on_clear).pack(
            side=tk.RIGHT
        )

        # Monte Carlo checkbox
        self.mc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            main, text="Run Monte Carlo validation (100k paths)", variable=self.mc_var
        ).pack(anchor=tk.W, pady=(0, 12))

        # Results section
        result_frame = ttk.LabelFrame(main, text="Results", padding=12)
        result_frame.pack(fill=tk.BOTH, expand=True)

        # Price display (prominent)
        self.price_var = tk.StringVar(value="—")
        price_row = ttk.Frame(result_frame)
        price_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(price_row, text="Option Price:", style="ResultTitle.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            price_row,
            textvariable=self.price_var,
            style="Result.TLabel",
            font=("Consolas", 14, "bold"),
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Greeks grid
        greeks_frame = ttk.Frame(result_frame)
        greeks_frame.pack(fill=tk.BOTH, expand=True)

        self.greek_vars = {}
        greek_labels = [
            ("Delta (Δ)", "delta"),
            ("Gamma (Γ)", "gamma"),
            ("Vega (ν, per 1%)", "vega"),
            ("Theta (Θ, per day)", "theta"),
            ("Rho (ρ, per 1%)", "rho"),
        ]

        for i, (label, key) in enumerate(greek_labels):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(greeks_frame, text=label + ":", width=18, style="ResultTitle.TLabel").grid(
                row=row, column=col, sticky=tk.W, padx=(0, 8), pady=4
            )
            var = tk.StringVar(value="—")
            self.greek_vars[key] = var
            ttk.Label(greeks_frame, textvariable=var, style="Result.TLabel").grid(
                row=row, column=col + 1, sticky=tk.W, padx=(0, 20), pady=4
            )

        greeks_frame.columnconfigure(1, weight=1)
        greeks_frame.columnconfigure(3, weight=1)

        # Monte Carlo result
        self.mc_var_label = tk.StringVar(value="")
        mc_row = ttk.Frame(result_frame)
        mc_row.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(mc_row, text="Monte Carlo:", style="ResultTitle.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            mc_row, textvariable=self.mc_var_label, style="Result.TLabel", foreground="#6b7280"
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(8, 4)
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _get_params(self) -> OptionParams | None:
        """Parse and validate input fields."""
        try:
            spot = float(self.entries["spot"].get())
            strike = float(self.entries["strike"].get())
            time = float(self.entries["time"].get())
            rate = float(self.entries["rate"].get())
            vol = float(self.entries["vol"].get())
            div_yield = float(self.entries["div_yield"].get())
            option_type = (
                OptionType.CALL if self.option_type_var.get() == "call" else OptionType.PUT
            )

            if spot <= 0 or strike <= 0 or time <= 0 or vol <= 0 or rate < 0 or div_yield < 0:
                raise ValueError("All values must be positive (rate/dividend can be zero)")

            return OptionParams(
                spot=spot,
                strike=strike,
                time_to_maturity=time,
                risk_free_rate=rate,
                volatility=vol,
                option_type=option_type,
                dividend_yield=div_yield,
            )
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return None

    def _on_price(self) -> None:
        """Price the option and display the Greeks."""
        params = self._get_params()
        if params is None:
            return

        self.status_var.set("Calculating...")
        self.root.update_idletasks()

        result = price_option(params)
        self.price_var.set(f"{result.price:.6f}")
        self._update_greeks(result.greeks)

        if self.mc_var.get():
            self.status_var.set("Running Monte Carlo (100k paths)...")
            self.root.update_idletasks()
            mc_price, mc_se = monte_carlo_price(params, n_paths=100_000, seed=0)
            diff = result.price - mc_price
            within = "OK" if abs(diff) <= 3 * mc_se else "OUT OF TOLERANCE"
            self.mc_var_label.set(f"MC={mc_price:.6f} +/- {mc_se:.6f} | Diff={diff:.6f} {within}")
        else:
            self.mc_var_label.set("")

        self.status_var.set("Done")

    def _on_iv(self) -> None:
        """Solve for the implied volatility of an observed market price."""
        params = self._get_params()
        if params is None:
            return

        dialog = MarketPriceDialog(self.root, "Implied Volatility", "Enter observed market price:")
        self.root.wait_window(dialog)

        if dialog.result is None:
            self.status_var.set("Implied volatility cancelled")
            return

        market_price = dialog.result

        self.status_var.set("Solving for implied volatility...")
        self.root.update_idletasks()

        try:
            iv = implied_volatility(market_price, params)
        except ValueError as exc:
            messagebox.showerror("Implied Volatility Error", str(exc))
            self.status_var.set("Implied volatility failed")
            return

        # Volatility is irrelevant to the inversion, so reuse the parsed
        # params directly rather than rebuilding them with a dummy value.
        self.entries["vol"].delete(0, tk.END)
        self.entries["vol"].insert(0, f"{iv:.6f}")
        messagebox.showinfo(
            "Implied Volatility",
            f"Implied Volatility: {iv:.6f} ({iv * 100:.2f}%)\nMarket Price: {market_price:.6f}",
        )
        self.status_var.set(f"Implied vol: {iv:.2%}")

    def _on_clear(self) -> None:
        """Clear all inputs and results."""
        defaults = {
            "spot": "100.0",
            "strike": "100.0",
            "time": "1.0",
            "rate": "0.05",
            "vol": "0.20",
            "div_yield": "0.0",
        }
        for key, value in defaults.items():
            self.entries[key].delete(0, tk.END)
            self.entries[key].insert(0, value)
        self.option_type_var.set("call")
        self.mc_var.set(False)
        self.price_var.set("—")
        for var in self.greek_vars.values():
            var.set("—")
        self.mc_var_label.set("")
        self.status_var.set("Cleared")

    def _update_greeks(self, greeks: Greeks) -> None:
        """Update Greek display."""
        self.greek_vars["delta"].set(f"{greeks.delta:.6f}")
        self.greek_vars["gamma"].set(f"{greeks.gamma:.6f}")
        self.greek_vars["vega"].set(f"{greeks.vega:.6f}")
        self.greek_vars["theta"].set(f"{greeks.theta:.6f}")
        self.greek_vars["rho"].set(f"{greeks.rho:.6f}")


class MarketPriceDialog(tk.Toplevel):
    """Simple dialog to get market price for IV calculation."""

    def __init__(self, parent: tk.Tk, title: str, prompt: str):
        super().__init__(parent)
        self.title(title)
        self.geometry("320x140")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: float | None = None

        ttk.Label(self, text=prompt, padding=(16, 16, 16, 8)).pack()
        self.entry = ttk.Entry(self, width=20)
        self.entry.pack(pady=(0, 16))
        self.entry.focus()

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=(0, 16))
        ttk.Button(btn_frame, text="OK", command=self._on_ok).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel).pack(side=tk.LEFT, padx=8)

        self.bind("<Return>", lambda e: self._on_ok())
        self.bind("<Escape>", lambda e: self._on_cancel())

    def _on_ok(self) -> None:
        try:
            self.result = float(self.entry.get())
            self.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")

    def _on_cancel(self) -> None:
        self.result = None
        self.destroy()


def run_gui() -> None:
    """Launch the GUI application."""
    root = tk.Tk()
    BlackScholesGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
