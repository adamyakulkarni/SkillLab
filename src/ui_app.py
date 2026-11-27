"""Simplified, interactive-map-centered Tkinter application for Team Infinite Loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.aqi_calculator import CPCB_CATEGORIES, get_aqi_category
from src.data_pipeline import (
    CleaningReport,
    load_and_clean_hourly_aqi,
    load_and_clean_pollutants,
    load_custom_csv,
)
from src.ml_models import (
    TrainingBundle,
    predict_single_scenario,
    train_cross_pollutant_models,
    train_next_day_aqi_models,
)
from src.seasonal_analysis import (
    BENGALURU_AREA_PROFILES,
    BENGALURU_STATION_COORDS,
    FIREWORK_HEAVY_METALS,
    SEASONAL_CAUSE_PROFILES,
    compute_diwali_window_analysis,
    compute_monthly_summary,
    compute_station_summary,
    generate_dynamic_insights,
)

NAVY = "#0F172A"
SLATE = "#1E293B"
ACCENT = "#0284C7"
BG = "#F1F5F9"
CARD = "#FFFFFF"
BORDER = "#CBD5E1"
TEXT = "#0F172A"
MUTED = "#475569"

DATASET_HOURLY = "Bengaluru Hourly AQI (2017-2023, 14 Areas)"
DATASET_POLLUTANTS = "Bengaluru Pollutant Data (2015-2018, 10 Areas)"

MONTH_LIST = [
    ("Full Year", None),
    ("Jan", 1), ("Feb", 2), ("Mar", 3), ("Apr", 4), ("May", 5), ("Jun", 6),
    ("Jul", 7), ("Aug", 8), ("Sep", 9), ("Oct", 10), ("Nov", 11), ("Dec", 12),
]

SEASON_PRESETS = [
    ("Winter (Dec-Feb)", [12, 1, 2], "Thermal inversions & morning/evening leaf/garbage burning trap PM2.5 & CO."),
    ("Summer Dust (Mar-May)", [3, 4, 5], "Dry roads, metro/building construction (PM10) & sunlight-driven Ozone (O3)."),
    ("Monsoon (Jun-Sep)", [6, 7, 8, 9], "Cleanest season! Heavy rain washes out PM2.5/PM10 & winds disperse gases."),
    ("Diwali / Festive (Oct-Nov)", [10, 11], "Fireworks release heavy metals (Ba, Sr, K, Cu, Al, Pb) & PM2.5 on calm nights."),
]

# Label offsets (dlon, dlat) in degrees to keep nearby Bengaluru pins readable
LABEL_OFFSETS: Dict[str, Tuple[float, float]] = {
    "Peenya": (0.004, 0.005),
    "Shivapura Peenya": (-0.005, -0.011),
    "Silk Board": (0.006, 0.003),
    "BTM Layout": (-0.022, -0.011),
    "BTM": (-0.020, -0.011),
    "SILKBOARD": (0.006, 0.003),
    "Jayanagar 5th Block": (-0.032, 0.004),
    "JAYANAGAR": (-0.028, 0.004),
    "Hombegowda Nagar": (0.004, 0.006),
    "NIHMANS": (0.004, 0.006),
    "Bapuji Nagar": (-0.028, -0.009),
    "KAVIKA": (0.004, -0.009),
}


class SimpleChart:
    """Lightweight Matplotlib figure embedded in a Tkinter frame."""

    def __init__(self, parent: tk.Widget, figsize: Tuple[float, float] = (6.5, 4.8), dpi: int = 100):
        self.frame = tk.Frame(parent, bg=CARD, bd=1, relief="solid")
        self.figure = Figure(figsize=figsize, dpi=dpi, facecolor=CARD)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def pack(self, **kwargs: Any) -> None:
        self.frame.pack(**kwargs)

    def draw(self, draw_fn: Callable[[Figure], None]) -> None:
        self.figure.clear()
        draw_fn(self.figure)
        self.figure.tight_layout(pad=1.4)
        self.canvas.draw_idle()


class AirQualityApp:
    """Clean 3-tab Bengaluru Air Quality Map & Pattern Explorer."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bengaluru Air Quality Map & Pattern Explorer | Team Infinite Loop")
        self.root.geometry("1280x820")
        self.root.minsize(1060, 680)
        self.root.configure(bg=BG)

        self.dataset_choice = tk.StringVar(value=DATASET_HOURLY)
        self.clean_df: Optional[pd.DataFrame] = None
        self.raw_df: Optional[pd.DataFrame] = None
        self.report: Optional[CleaningReport] = None
        self.metric_col: str = "Daily_Mean_AQI"
        self.training_bundle: Optional[TrainingBundle] = None

        # Map interactive state
        self.selected_period_label = tk.StringVar(value="Full Year")
        self.selected_months: Optional[List[int]] = None
        self.map_agg_mode = tk.StringVar(value="Mean AQI")
        self.selected_station = tk.StringVar(value="City Railway Station")
        self.station_screen_points: List[Tuple[str, float, float]] = []
        self._animating = False
        self._anim_job: Optional[str] = None
        self._anim_month = 1

        self._setup_styles()
        self._build_header()
        self._build_tabs()
        self._build_status_bar()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, lambda: self.load_dataset(DATASET_HOURLY))

    def _on_close(self) -> None:
        self._stop_animation()
        self.root.destroy()

    def _setup_styles(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(14, 8), background="#E2E8F0", foreground=TEXT)
        style.map("TNotebook.Tab", background=[("selected", CARD)], foreground=[("selected", "#0369A1")])
        style.configure("Treeview", font=("Segoe UI", 9), rowheight=24)
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

    def _build_header(self) -> None:
        hdr = tk.Frame(self.root, bg=NAVY, padx=16, pady=10)
        hdr.pack(fill="x")

        left = tk.Frame(hdr, bg=NAVY)
        left.pack(side="left", fill="x", expand=True)
        tk.Label(
            left,
            text="BENGALURU AIR QUALITY MAP & PATTERN EXPLORER",
            font=("Segoe UI", 14, "bold"),
            bg=NAVY,
            fg="#F8FAFC",
        ).pack(anchor="w")
        tk.Label(
            left,
            text="Team Infinite Loop  •  Nilasha B • Abdulhai Nalband • Adamya J Kulkarni • Aditya S Lokare",
            font=("Segoe UI", 9),
            bg=NAVY,
            fg="#94A3B8",
        ).pack(anchor="w", pady=(2, 0))

        right = tk.Frame(hdr, bg=NAVY)
        right.pack(side="right")
        tk.Label(right, text="Dataset:", font=("Segoe UI", 9, "bold"), bg=NAVY, fg="#CBD5E1").pack(side="left", padx=(0, 6))
        cb = ttk.Combobox(
            right,
            textvariable=self.dataset_choice,
            values=[DATASET_HOURLY, DATASET_POLLUTANTS],
            state="readonly",
            width=40,
        )
        cb.pack(side="left", padx=(0, 8))
        cb.bind("<<ComboboxSelected>>", lambda _e: self.load_dataset(self.dataset_choice.get()))

        tk.Button(
            right,
            text="Load Custom CSV...",
            font=("Segoe UI", 9, "bold"),
            bg=SLATE,
            fg="#F8FAFC",
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._on_browse_csv,
        ).pack(side="left")

    def _build_tabs(self) -> None:
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=8)

        self.tab_map = tk.Frame(self.nb, bg=BG)
        self.tab_causes = tk.Frame(self.nb, bg=BG)
        self.tab_data_ml = tk.Frame(self.nb, bg=BG)

        self.nb.add(self.tab_map, text="1. Interactive Bengaluru Map")
        self.nb.add(self.tab_causes, text="2. Worst Times, Heavy Metals & Causes")
        self.nb.add(self.tab_data_ml, text="3. Data Cleaning & Simple Predictor")

        self._init_tab_map()
        self._init_tab_causes()
        self._init_tab_data_ml()

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self.root, bg="#E2E8F0", padx=12, pady=4)
        bar.pack(fill="x", side="bottom")
        self.status_var = tk.StringVar(value="Loading Bengaluru air quality data...")
        tk.Label(bar, textvariable=self.status_var, font=("Segoe UI", 9), bg="#E2E8F0", fg=TEXT).pack(side="left")
        tk.Label(bar, text="Python • Pandas • NumPy • Matplotlib • Scikit-learn", font=("Segoe UI", 9), bg="#E2E8F0", fg=MUTED).pack(side="right")

    def _make_card(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg=CARD, bd=1, relief="solid")
        hdr = tk.Frame(outer, bg="#F8FAFC", padx=10, pady=6)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, font=("Segoe UI", 10, "bold"), bg="#F8FAFC", fg=TEXT).pack(anchor="w")
        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x")
        body = tk.Frame(outer, bg=CARD, padx=10, pady=8)
        body.pack(fill="both", expand=True)
        return body

    # ------------------------------------------------------------------
    # Dataset Loading
    # ------------------------------------------------------------------
    def load_dataset(self, key: str, custom_path: Optional[Path] = None) -> None:
        self._stop_animation()
        self.status_var.set(f"Loading {key}...")
        self.root.update_idletasks()
        try:
            if custom_path is not None:
                raw_df, clean_df, report = load_custom_csv(custom_path)
                self.metric_col = "AQI"
            elif key == DATASET_HOURLY:
                raw_df, clean_df, report = load_and_clean_hourly_aqi()
                self.metric_col = "Daily_Mean_AQI"
            else:
                raw_df, clean_df, report = load_and_clean_pollutants()
                self.metric_col = "AQI"
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc))
            return

        self.raw_df = raw_df
        self.clean_df = clean_df
        self.report = report
        self.training_bundle = None
        self.selected_period_label.set("Full Year")
        self.selected_months = None
        self.period_banner.configure(text="Showing: Full Year")
        for btn in self.month_buttons.values():
            btn.configure(bg="#F1F5F9", fg=TEXT)

        stations = sorted(clean_df["Station"].dropna().unique().tolist())
        if stations and self.selected_station.get() not in stations:
            self.selected_station.set(stations[0])
        self.pred_station_cb["values"] = stations
        if stations:
            self.pred_station_var.set(stations[0])

        self._refresh_map_and_insights()
        self._refresh_causes_tab()
        self._refresh_cleaning_tab()
        self._train_simple_model_async()

        self.status_var.set(
            f"Loaded {report.cleaned_rows:,} cleaned records across {len(report.stations)} Bengaluru areas "
            f"({report.date_start} to {report.date_end}). Click any area pin on the map to inspect."
        )

    def _on_browse_csv(self) -> None:
        fp = filedialog.askopenfilename(title="Open Air Quality CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if fp:
            self.load_dataset("Custom CSV", Path(fp))

    # ------------------------------------------------------------------
    # TAB 1: Interactive Bengaluru Map
    # ------------------------------------------------------------------
    def _init_tab_map(self) -> None:
        # Top Time-of-Year Control Bar
        ctrl = tk.Frame(self.tab_map, bg=CARD, bd=1, relief="solid", padx=10, pady=8)
        ctrl.pack(fill="x", padx=8, pady=(8, 4))

        row1 = tk.Frame(ctrl, bg=CARD)
        row1.pack(fill="x", pady=(0, 5))
        tk.Label(row1, text="Season Quick-Select:", font=("Segoe UI", 9, "bold"), bg=CARD, fg=TEXT).pack(side="left", padx=(0, 6))

        tk.Button(
            row1, text="Full Year", font=("Segoe UI", 9, "bold"), bg="#E0F2FE", fg="#0369A1",
            relief="flat", padx=8, pady=3, cursor="hand2",
            command=lambda: self._set_period("Full Year", None),
        ).pack(side="left", padx=2)

        for label, months, _reason in SEASON_PRESETS:
            tk.Button(
                row1, text=label, font=("Segoe UI", 9, "bold"), bg="#E2E8F0", fg=TEXT,
                relief="flat", padx=8, pady=3, cursor="hand2",
                command=lambda lbl=label, ms=months: self._set_period(lbl, ms),
            ).pack(side="left", padx=2)

        self.play_btn = tk.Button(
            row1, text="Play Year (Jan-Dec)", font=("Segoe UI", 9, "bold"), bg="#0F766E", fg="#FFFFFF",
            relief="flat", padx=10, pady=3, cursor="hand2", command=self._toggle_animation,
        )
        self.play_btn.pack(side="right", padx=(6, 0))

        ttk.Combobox(
            row1, textvariable=self.map_agg_mode, values=["Mean AQI", "Peak Days (95th Pct)"],
            state="readonly", width=20,
        ).pack(side="right", padx=6)
        self.map_agg_mode.trace_add("write", lambda *_: self._refresh_map_and_insights())

        row2 = tk.Frame(ctrl, bg=CARD)
        row2.pack(fill="x")
        tk.Label(row2, text="Or Pick Month:", font=("Segoe UI", 9, "bold"), bg=CARD, fg=MUTED).pack(side="left", padx=(0, 6))
        self.month_buttons: Dict[str, tk.Button] = {}
        for m_label, m_num in MONTH_LIST[1:]:
            b = tk.Button(
                row2, text=m_label, font=("Segoe UI", 9), bg="#F1F5F9", fg=TEXT,
                relief="flat", padx=8, pady=2, cursor="hand2",
                command=lambda lbl=m_label, num=m_num: self._set_period(lbl, [num] if num else None),
            )
            b.pack(side="left", padx=2)
            self.month_buttons[m_label] = b

        self.period_banner = tk.Label(
            row2, text="Showing: Full Year", font=("Segoe UI", 9, "bold"), bg=CARD, fg=ACCENT,
        )
        self.period_banner.pack(side="right", padx=6)

        # Main Split: Left Map (60%) | Right Insights & Ranked Areas (40%)
        pane = tk.PanedWindow(self.tab_map, orient="horizontal", bg=BG, sashwidth=6)
        pane.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        left = tk.Frame(pane, bg=BG)
        right = tk.Frame(pane, bg=BG)
        pane.add(left, stretch="always", minsize=620)
        pane.add(right, stretch="always", minsize=400)

        self.map_chart = SimpleChart(left, figsize=(7.0, 5.8), dpi=100)
        self.map_chart.pack(fill="both", expand=True)
        self.map_chart.canvas.mpl_connect("button_press_event", self._on_map_click)

        # Right Top: Selected Area Details
        area_card = self._make_card(right, "Selected Bengaluru Area (Click any pin on the map)")
        area_card.master.pack(fill="x", pady=(0, 6))

        self.area_badge = tk.Frame(area_card, bg="#16A34A", padx=10, pady=8)
        self.area_badge.pack(fill="x", pady=(0, 6))
        self.area_title_lbl = tk.Label(self.area_badge, text="—", font=("Segoe UI", 12, "bold"), bg="#16A34A", fg="#FFFFFF", anchor="w")
        self.area_title_lbl.pack(fill="x")
        self.area_sub_lbl = tk.Label(self.area_badge, text="—", font=("Segoe UI", 9), bg="#16A34A", fg="#FFFFFF", anchor="w")
        self.area_sub_lbl.pack(fill="x")

        self.area_details_lbl = tk.Label(
            area_card, text="", font=("Segoe UI", 9), bg=CARD, fg=TEXT, justify="left", wraplength=410, anchor="w"
        )
        self.area_details_lbl.pack(fill="x")

        # Right Middle: Season Reason Callout
        season_card = self._make_card(right, "Why Air Quality Changes at This Time of Year")
        season_card.master.pack(fill="x", pady=(0, 6))
        self.season_reason_lbl = tk.Label(
            season_card, text="", font=("Segoe UI", 9), bg=CARD, fg=TEXT, justify="left", wraplength=410, anchor="w"
        )
        self.season_reason_lbl.pack(fill="x")

        # Right Bottom: Ranked Areas Table
        rank_card = self._make_card(right, "Areas Ranked Worst to Cleanest (Click row to locate)")
        rank_card.master.pack(fill="both", expand=True)
        self.map_rank_tree = ttk.Treeview(rank_card, columns=("area", "aqi", "cat"), show="headings", height=8)
        for col, txt, w in [("area", "Bengaluru Area", 185), ("aqi", "AQI", 65), ("cat", "Category", 145)]:
            self.map_rank_tree.heading(col, text=txt)
            self.map_rank_tree.column(col, width=w, anchor="center" if col == "aqi" else "w")
        ysb = ttk.Scrollbar(rank_card, orient="vertical", command=self.map_rank_tree.yview)
        self.map_rank_tree.configure(yscrollcommand=ysb.set)
        self.map_rank_tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        self.map_rank_tree.bind("<<TreeviewSelect>>", self._on_rank_row_select)

    def _set_period(self, label: str, months: Optional[List[int]]) -> None:
        self.selected_period_label.set(label)
        self.selected_months = months
        self.period_banner.configure(text=f"Showing: {label}")
        for m_lbl, btn in self.month_buttons.items():
            btn.configure(bg=ACCENT if m_lbl == label else "#F1F5F9", fg="#FFFFFF" if m_lbl == label else TEXT)
        self._refresh_map_and_insights()

    def _toggle_animation(self) -> None:
        if self._animating:
            self._stop_animation()
        else:
            self._animating = True
            self._anim_month = 1
            self.play_btn.configure(text="Stop Animation", bg="#DC2626")
            self._step_animation()

    def _stop_animation(self) -> None:
        self._animating = False
        if self._anim_job is not None:
            try:
                self.root.after_cancel(self._anim_job)
            except Exception:
                pass
            self._anim_job = None
        if hasattr(self, "play_btn"):
            self.play_btn.configure(text="Play Year (Jan-Dec)", bg="#0F766E")

    def _step_animation(self) -> None:
        if not self._animating:
            return
        m_label, m_num = MONTH_LIST[self._anim_month]
        self._set_period(m_label, [m_num] if m_num else None)
        self._anim_month = (self._anim_month % 12) + 1
        self._anim_job = self.root.after(1100, self._step_animation)

    def _period_filtered_df(self) -> pd.DataFrame:
        assert self.clean_df is not None
        if self.selected_months:
            return self.clean_df[self.clean_df["Month"].isin(self.selected_months)]
        return self.clean_df

    def _on_map_click(self, event: Any) -> None:
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        best_station = None
        best_dist = float("inf")
        for st, lon, lat in self.station_screen_points:
            d = (event.xdata - lon) ** 2 + (event.ydata - lat) ** 2
            if d < best_dist:
                best_dist = d
                best_station = st
        # Select if clicked within reasonable radius (~0.025 deg)
        if best_station is not None and best_dist <= 0.025 ** 2:
            self.selected_station.set(best_station)
            self._refresh_map_and_insights()

    def _on_rank_row_select(self, _e: Any) -> None:
        sel = self.map_rank_tree.selection()
        if not sel:
            return
        area = str(self.map_rank_tree.item(sel[0])["values"][0])
        if area != self.selected_station.get():
            self.selected_station.set(area)
            self._refresh_map_and_insights()

    def _refresh_map_and_insights(self) -> None:
        if self.clean_df is None:
            return
        df_period = self._period_filtered_df()
        metric = self.metric_col
        use_p95 = "Peak" in self.map_agg_mode.get()

        if use_p95:
            st_stats = (
                df_period.groupby("Station")[metric]
                .agg(val=lambda s: float(np.nanpercentile(s.dropna(), 95)) if len(s.dropna()) else np.nan, n="count")
                .reset_index()
            )
        else:
            st_stats = (
                df_period.groupby("Station")[metric]
                .agg(val="mean", n="count")
                .reset_index()
            )
        st_stats = st_stats.dropna(subset=["val"]).sort_values("val", ascending=False).reset_index(drop=True)

        # Update ranked table
        for item in self.map_rank_tree.get_children():
            self.map_rank_tree.delete(item)
        for _, r in st_stats.iterrows():
            cat = get_aqi_category(r["val"])
            self.map_rank_tree.insert("", "end", values=(r["Station"], f"{r['val']:.1f}", cat.label))

        # Update Selected Area Card
        sel_st = self.selected_station.get()
        row_match = st_stats[st_stats["Station"] == sel_st]
        if row_match.empty and not st_stats.empty:
            sel_st = str(st_stats.iloc[0]["Station"])
            self.selected_station.set(sel_st)
            row_match = st_stats.iloc[[0]]

        if not row_match.empty:
            val = float(row_match.iloc[0]["val"])
            cat = get_aqi_category(val)
            prof = BENGALURU_AREA_PROFILES.get(sel_st, {"zone": "Bengaluru Urban Station", "drivers": "Urban traffic & seasonal inversions."})

            # Station-specific worst & cleanest months
            st_full = self.clean_df[self.clean_df["Station"] == sel_st]
            m_df = compute_monthly_summary(st_full, metric)
            worst_m = ", ".join(f"{r['Month_Name']} ({r['Mean']:.0f})" for _, r in m_df.sort_values("Mean", ascending=False).head(3).iterrows())
            clean_m = ", ".join(f"{r['Month_Name']} ({r['Mean']:.0f})" for _, r in m_df.sort_values("Mean", ascending=True).head(2).iterrows())

            self.area_badge.configure(bg=cat.color)
            self.area_title_lbl.configure(text=f"{sel_st}  —  AQI {val:.1f} ({cat.label})", bg=cat.color, fg=cat.text_color)
            self.area_sub_lbl.configure(text=f"{prof['zone']}  •  Period: {self.selected_period_label.get()}", bg=cat.color, fg=cat.text_color)
            self.area_details_lbl.configure(
                text=(
                    f"Worst Months Here: {worst_m}\n"
                    f"Cleanest Months Here: {clean_m}\n"
                    f"Why Pollution Builds Up Here: {prof['drivers']}\n"
                    f"Health Note: {cat.advisory}"
                )
            )

        # Update Season Reason Card
        lbl = self.selected_period_label.get()
        reason_txt = (
            "Across the year, Bengaluru's air is cleanest in Monsoon (Jun-Sep, ~55-59 AQI) and worst in "
            "Winter (Dec-Feb, ~90 AQI), Dry Pre-Monsoon (Mar-Apr, ~86-92 AQI), and Diwali week (Oct-Nov, +38.6% surge)."
        )
        for s_lbl, s_months, s_reason in SEASON_PRESETS:
            if lbl == s_lbl or (self.selected_months and len(self.selected_months) == 1 and self.selected_months[0] in s_months):
                reason_txt = f"{s_lbl}: {s_reason}"
                break
        self.season_reason_lbl.configure(text=reason_txt)

        # Draw Bengaluru Geographic Map
        self._draw_bengaluru_map(st_stats, sel_st)

    def _draw_bengaluru_map(self, st_stats: pd.DataFrame, sel_st: str) -> None:
        self.station_screen_points = []
        period_lbl = self.selected_period_label.get()
        agg_lbl = self.map_agg_mode.get()

        def _draw(fig: Figure) -> None:
            ax = fig.add_subplot(111)
            ax.set_facecolor("#EEF2F6")

            # Subtle Bengaluru Geographic Context (Outer Ring Road, Arterial Roads, Lakes)
            orr_lon = [77.5988, 77.6530, 77.6983, 77.6210, 77.5842, 77.5250, 77.5197, 77.5988]
            orr_lat = [13.0354, 13.0040, 12.9473, 12.9170, 12.9150, 12.9480, 13.0285, 13.0354]
            ax.plot(orr_lon, orr_lat, color="#CBD5E1", linewidth=2.5, linestyle="-", zorder=1, label="Outer Ring Road (approx.)")

            # Major corridors: Mysuru Rd, Bellary/Airport Rd, Tumakuru Rd, Hosur Rd, Old Madras Rd
            corridors = [
                ([77.5713, 77.5988, 77.6000], [12.9767, 13.0354, 13.0700]),  # Bellary / Airport Rd
                ([77.5713, 77.5197, 77.4900], [12.9767, 13.0285, 13.0600]),  # Tumakuru Rd (Peenya)
                ([77.5713, 77.5397, 77.5030, 77.4800], [12.9767, 12.9568, 12.9250, 12.9050]),  # Mysuru Rd
                ([77.5713, 77.6210, 77.6350], [12.9767, 12.9170, 12.7850]),  # Hosur Rd (Silk Board -> Jigani)
                ([77.5713, 77.6530, 77.7200], [12.9767, 13.0040, 13.0100]),  # Old Madras Rd
            ]
            for clons, clats in corridors:
                ax.plot(clons, clats, color="#CBD5E1", linewidth=1.3, linestyle="--", zorder=1)

            # Subtle green park / blue lake landmarks for spatial orientation
            ax.scatter([77.585, 77.593], [12.951, 12.976], s=320, color="#BBF7D0", alpha=0.65, zorder=1)  # Lalbagh / Cubbon
            ax.scatter([77.670, 77.619], [12.936, 12.982], s=260, color="#BFDBFE", alpha=0.7, zorder=1)  # Bellandur / Ulsoor Lakes
            ax.text(77.593, 12.984, "Central Bengaluru", fontsize=7.5, color="#64748B", ha="center", style="italic", zorder=2)

            # Plot stations
            for _, r in st_stats.iterrows():
                st = str(r["Station"])
                if st not in BENGALURU_STATION_COORDS:
                    continue
                lat, lon = BENGALURU_STATION_COORDS[st]
                val = float(r["val"])
                cat = get_aqi_category(val)
                size = max(130, min(val * 3.2, 620))
                is_sel = (st == sel_st)

                self.station_screen_points.append((st, lon, lat))

                if is_sel:
                    ax.scatter([lon], [lat], s=size + 260, facecolors="none", edgecolors="#0F172A", linewidths=2.4, zorder=4)

                ax.scatter(
                    [lon], [lat], s=size, color=cat.color,
                    edgecolors="#0F172A" if is_sel else "#FFFFFF", linewidths=1.5, alpha=0.92, zorder=5,
                )

                dlon, dlat = LABEL_OFFSETS.get(st, (0.004, 0.004))
                short_name = st.replace(" Kadabesanahalli", " (ORR)").replace(" 5th Block", "")
                ax.text(
                    lon + dlon, lat + dlat, f"{short_name}\n({val:.0f})",
                    fontsize=7.5, fontweight="bold" if is_sel else "normal",
                    color="#0F172A",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor=cat.color if is_sel else "#CBD5E1", alpha=0.88),
                    zorder=6,
                )

            ax.set_xlim(77.465, 77.745)
            ax.set_ylim(12.775, 13.065)
            ax.set_xlabel("Longitude (E)", fontsize=8.5, color=MUTED)
            ax.set_ylabel("Latitude (N)", fontsize=8.5, color=MUTED)
            ax.set_title(f"Bengaluru Air Quality Map — {period_lbl} ({agg_lbl})  [Click any area]", fontsize=10.5, fontweight="bold", color=TEXT)
            ax.grid(True, linestyle=":", alpha=0.45, color="#94A3B8")

            # Compact CPCB AQI Color Legend
            legend_items = [
                Line2D([0], [0], marker="o", color="w", label=f"{c.label} ({c.min_aqi}-{c.max_aqi})", markerfacecolor=c.color, markersize=8)
                for c in CPCB_CATEGORIES[:5]
            ]
            ax.legend(handles=legend_items, loc="lower left", fontsize=7.5, framealpha=0.92, title="CPCB AQI Scale", title_fontsize=8)

        self.map_chart.draw(_draw)

    # ------------------------------------------------------------------
    # TAB 2: Worst Times of the Year, Heavy Metals & Causes
    # ------------------------------------------------------------------
    def _init_tab_causes(self) -> None:
        self.causes_banner = tk.Label(
            self.tab_causes, text="", font=("Segoe UI", 10, "bold"),
            bg="#FEF2F2", fg="#991B1B", padx=12, pady=7, anchor="w", wraplength=1220, justify="left",
        )
        self.causes_banner.pack(fill="x", padx=8, pady=(8, 4))

        pane = tk.PanedWindow(self.tab_causes, orient="horizontal", bg=BG, sashwidth=6)
        pane.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = tk.Frame(pane, bg=BG)
        right = tk.Frame(pane, bg=BG)
        pane.add(left, stretch="always", minsize=540)
        pane.add(right, stretch="always", minsize=500)

        # Left: Two stacked clean charts (Monthly Curve + Diwali Spikes)
        self.monthly_chart = SimpleChart(left, figsize=(6.0, 2.7), dpi=95)
        self.monthly_chart.pack(fill="both", expand=True, pady=(0, 6))

        self.diwali_chart = SimpleChart(left, figsize=(6.0, 2.6), dpi=95)
        self.diwali_chart.pack(fill="both", expand=True)

        # Right: Clear structured explanations
        c1 = self._make_card(right, "When Air Quality Is Particularly Bad & Why")
        c1.master.pack(fill="x", pady=(0, 6))
        for prof in SEASONAL_CAUSE_PROFILES:
            box = tk.Frame(c1, bg="#F8FAFC", bd=1, relief="solid", padx=8, pady=5)
            box.pack(fill="x", pady=2)
            tk.Label(box, text=f"{prof['season']}  |  {prof['severity']}", font=("Segoe UI", 9, "bold"), bg="#F8FAFC", fg="#0369A1").pack(anchor="w")
            tk.Label(box, text=prof["why_it_happens"], font=("Segoe UI", 9), bg="#F8FAFC", fg=TEXT, wraplength=540, justify="left").pack(anchor="w")

        c2 = self._make_card(right, "Festival Fireworks (Diwali): Heavy Metals Inside PM2.5 Spikes")
        c2.master.pack(fill="both", expand=True)
        hm_tree = ttk.Treeview(c2, columns=("metal", "source", "impact"), show="headings", height=6)
        for col, txt, w in [("metal", "Heavy Metal", 120), ("source", "Firecracker Source", 200), ("impact", "Why It Harms Air Quality", 230)]:
            hm_tree.heading(col, text=txt)
            hm_tree.column(col, width=w, anchor="w")
        for row in FIREWORK_HEAVY_METALS:
            hm_tree.insert("", "end", values=(row["element"], row["source"], row["impact"]))
        hm_tree.pack(fill="both", expand=True)

    def _refresh_causes_tab(self) -> None:
        if self.clean_df is None:
            return
        insights = generate_dynamic_insights(self.clean_df, self.metric_col)
        self.causes_banner.configure(
            text=(
                f"WORST MONTHS: {insights['worst_months']}   •   CLEANEST MONTHS: {insights['cleanest_months']}   •   "
                f"TOP POLLUTED AREA: {insights['top_station']}   •   {insights['diwali_summary']}"
            )
        )

        df = self.clean_df
        metric = self.metric_col

        def _draw_monthly(fig: Figure) -> None:
            ax = fig.add_subplot(111)
            m_df = compute_monthly_summary(df, metric)
            x = np.arange(len(m_df))
            ax.axvspan(-0.5, 1.5, color="#DBEAFE", alpha=0.5, label="Winter Inversion")
            ax.axvspan(1.5, 4.5, color="#FEF3C7", alpha=0.5, label="Dry Pre-Monsoon Dust")
            ax.axvspan(4.5, 8.5, color="#DCFCE7", alpha=0.5, label="Monsoon Washout")
            ax.axvspan(8.5, 10.5, color="#FEE2E2", alpha=0.6, label="Diwali / Post-Monsoon")
            ax.axvspan(10.5, 11.5, color="#DBEAFE", alpha=0.5)
            ax.plot(x, m_df["Mean"], "o-", color="#0F172A", linewidth=2.2, label="Monthly Mean")
            for xi, v in zip(x, m_df["Mean"]):
                ax.text(xi, v + 2.2, f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(m_df["Month_Name"], fontsize=8.5)
            ax.set_ylabel(metric, fontsize=8.5)
            ax.set_title("Bengaluru Monthly Pattern (Worst in Nov-Mar, Cleanest in Jun-Sep)", fontsize=9.5, fontweight="bold")
            ax.legend(fontsize=7.5, ncol=3, loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.5)

        def _draw_diwali(fig: Figure) -> None:
            ax = fig.add_subplot(111)
            d_df = compute_diwali_window_analysis(df, metric)
            if d_df.empty:
                ax.text(0.5, 0.5, "No Diwali dates in dataset.", ha="center", va="center")
                return
            x = np.arange(len(d_df))
            w = 0.36
            ax.bar(x - w / 2, d_df["Month_Baseline_Mean"], width=w, color="#64748B", label="Normal Month Baseline")
            bars = ax.bar(x + w / 2, d_df["Festival_Window_Mean"], width=w, color="#DC2626", label="Diwali Days (D-1 to D+2)")
            for b, pct in zip(bars, d_df["Surge_vs_Baseline_Pct"]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{pct:+.0f}%", ha="center", fontsize=7.5, fontweight="bold", color="#991B1B")
            ax.set_xticks(x)
            ax.set_xticklabels([f"{int(r['Year'])}" for _, r in d_df.iterrows()], fontsize=8.5)
            ax.set_ylabel(metric, fontsize=8.5)
            ax.set_ylim(0, max(d_df["Festival_Window_Mean"].max() * 1.25, 150))
            ax.set_title("Diwali Fireworks Spike vs. Same-Month Baseline (Real CPCB Data)", fontsize=9.5, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle=":", alpha=0.5)

        self.monthly_chart.draw(_draw_monthly)
        self.diwali_chart.draw(_draw_diwali)

    # ------------------------------------------------------------------
    # TAB 3: Data Cleaning (Slides 06-07) & Simple Predictor (Slide 09)
    # ------------------------------------------------------------------
    def _init_tab_data_ml(self) -> None:
        pane = tk.PanedWindow(self.tab_data_ml, orient="horizontal", bg=BG, sashwidth=6)
        pane.pack(fill="both", expand=True, padx=8, pady=8)

        left = tk.Frame(pane, bg=BG)
        right = tk.Frame(pane, bg=BG)
        pane.add(left, stretch="always", minsize=520)
        pane.add(right, stretch="always", minsize=500)

        # Left: Slide 06-07 Data Cleaning Proof
        clean_card = self._make_card(left, "Data Preparation & Cleaning (Slides 06-07: Pandas Pipeline)")
        clean_card.master.pack(fill="both", expand=True)
        self.clean_summary_lbl = tk.Label(
            clean_card, text="", font=("Segoe UI", 9), bg="#F8FAFC", fg=TEXT, justify="left", wraplength=510, padx=8, pady=6
        )
        self.clean_summary_lbl.pack(fill="x", pady=(0, 8))
        self.clean_diag_chart = SimpleChart(clean_card, figsize=(5.5, 4.2), dpi=95)
        self.clean_diag_chart.pack(fill="both", expand=True)

        # Right: Slide 09 Simple Regression Predictor
        ml_card = self._make_card(right, "Simple AQI Regression Model & Estimator (Slide 09)")
        ml_card.master.pack(fill="both", expand=True)

        self.ml_metrics_lbl = tk.Label(
            ml_card, text="Training regression models...", font=("Segoe UI", 9), bg="#E0F2FE", fg="#0369A1",
            justify="left", wraplength=500, padx=8, pady=6,
        )
        self.ml_metrics_lbl.pack(fill="x", pady=(0, 8))

        form = tk.Frame(ml_card, bg=CARD)
        form.pack(fill="x", pady=(0, 6))
        self.pred_station_var = tk.StringVar(value="Silk Board")
        self.pred_month_var = tk.StringVar(value="11 - Nov (Diwali / Festive)")
        self.pred_prev_var = tk.DoubleVar(value=115.0)

        tk.Label(form, text="Bengaluru Area:", font=("Segoe UI", 9, "bold"), bg=CARD).grid(row=0, column=0, sticky="w", pady=4)
        self.pred_station_cb = ttk.Combobox(form, textvariable=self.pred_station_var, state="readonly", width=26)
        self.pred_station_cb.grid(row=0, column=1, sticky="w", padx=8, pady=4)

        tk.Label(form, text="Time of Year:", font=("Segoe UI", 9, "bold"), bg=CARD).grid(row=1, column=0, sticky="w", pady=4)
        months_ui = [
            "1 - Jan (Winter)", "2 - Feb (Winter)", "3 - Mar (Dry Dust)", "4 - Apr (Dry Dust)",
            "5 - May (Pre-Monsoon)", "6 - Jun (Monsoon)", "7 - Jul (Monsoon)", "8 - Aug (Monsoon)",
            "9 - Sep (Late Monsoon)", "10 - Oct (Diwali/Festive)", "11 - Nov (Diwali/Festive)", "12 - Dec (Winter)",
        ]
        ttk.Combobox(form, textvariable=self.pred_month_var, values=months_ui, state="readonly", width=26).grid(row=1, column=1, sticky="w", padx=8, pady=4)

        self.pred_input_lbl = tk.Label(form, text="Yesterday's AQI:", font=("Segoe UI", 9, "bold"), bg=CARD)
        self.pred_input_lbl.grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(form, textvariable=self.pred_prev_var, width=12).grid(row=2, column=1, sticky="w", padx=8, pady=4)

        tk.Button(
            ml_card, text="Estimate AQI", font=("Segoe UI", 9, "bold"), bg="#0F766E", fg="#FFFFFF",
            relief="flat", padx=14, pady=5, cursor="hand2", command=self._on_simple_predict,
        ).pack(anchor="w", pady=(2, 8))

        self.simple_pred_badge = tk.Label(
            ml_card, text="Estimated AQI will appear here.", font=("Segoe UI", 10, "bold"),
            bg="#E2E8F0", fg=TEXT, padx=10, pady=8, wraplength=490, justify="left",
        )
        self.simple_pred_badge.pack(fill="x", pady=(0, 8))

        self.ml_scatter_chart = SimpleChart(ml_card, figsize=(5.2, 2.7), dpi=92)
        self.ml_scatter_chart.pack(fill="both", expand=True)

    def _refresh_cleaning_tab(self) -> None:
        if self.clean_df is None or self.report is None:
            return
        rep = self.report
        examples = " | ".join(rep.glitch_examples[:3]) if rep.glitch_examples else "Out-of-range values clipped to [0, 500]"
        self.clean_summary_lbl.configure(
            text=(
                f"Dataset: {rep.dataset_name}\n"
                f"01 Load -> 02 Inspect -> 03 Clean -> 04 Prepare:\n"
                f"• Raw Shape: {rep.raw_rows:,} rows × {rep.raw_cols} columns   |   Cleaned Rows: {rep.cleaned_rows:,}\n"
                f"• Missing Cells: {rep.raw_missing_total:,} -> {rep.cleaned_missing_total:,}   |   Duplicates Checked: {rep.raw_duplicates:,}\n"
                f"• Sensor Glitches Filtered: {rep.sensor_glitches_flagged:,} ({examples})"
            )
        )

        df_c = self.clean_df
        def _draw_diag(fig: Figure) -> None:
            ax = fig.add_subplot(111)
            if "PM25_Raw" in df_c.columns and "PM25" in df_c.columns:
                raw_m = df_c.groupby("Month")["PM25_Raw"].mean()
                cln_m = df_c.groupby("Month")["PM25"].mean()
                m_idx = sorted(cln_m.index)
                ax.plot(m_idx, [raw_m.get(m, np.nan) for m in m_idx], "o--", color="#DC2626", label="Raw Uncleaned PM2.5 (Glitch Spikes)")
                ax.plot(m_idx, [cln_m.get(m, np.nan) for m in m_idx], "s-", color="#0284C7", linewidth=2.2, label="Cleaned PM2.5 (True Pattern)")
                ax.set_xticks(m_idx)
                ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(m_idx)], fontsize=8)
                ax.set_ylabel("PM2.5 (ug/m3)", fontsize=8.5)
                ax.set_title("Before vs. After Data Cleaning", fontsize=9.5, fontweight="bold")
                ax.legend(fontsize=8)
                ax.grid(True, linestyle=":", alpha=0.5)
            else:
                ax.hist(df_c["Daily_Mean_AQI"].dropna(), bins=35, color="#0284C7", alpha=0.8)
                ax.axvline(df_c["Daily_Mean_AQI"].mean(), color="#DC2626", linestyle="--", label=f"Mean = {df_c['Daily_Mean_AQI'].mean():.1f}")
                ax.set_xlabel("Cleaned Daily Mean AQI", fontsize=8.5)
                ax.set_ylabel("Number of Station-Days", fontsize=8.5)
                ax.set_title("Cleaned Bengaluru Daily AQI Distribution (2017-2023)", fontsize=9.5, fontweight="bold")
                ax.legend(fontsize=8)
                ax.grid(True, linestyle=":", alpha=0.5)

        self.clean_diag_chart.draw(_draw_diag)

    def _train_simple_model_async(self) -> None:
        if self.clean_df is None or self.report is None:
            return
        df_snap = self.clean_df.copy()
        is_hourly = self.report.dataset_type == "hourly_aqi"
        self.pred_input_lbl.configure(text="Yesterday's AQI:" if is_hourly else "NO2 Level (ug/m3):")
        self.pred_prev_var.set(115.0 if is_hourly else 45.0)
        self.ml_metrics_lbl.configure(text="Training Linear Regression & Random Forest models on cleaned Bengaluru data...")

        def _worker() -> None:
            try:
                if is_hourly:
                    bundle = train_next_day_aqi_models(df_snap, test_ratio=0.2, chronological_split=True)
                else:
                    bundle = train_cross_pollutant_models(df_snap, target_col="AQI", test_ratio=0.2)
                self.root.after(0, lambda: self._on_simple_model_ready(bundle))
            except Exception as exc:
                self.root.after(0, lambda: self.ml_metrics_lbl.configure(text=f"Model training note: {exc}"))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_simple_model_ready(self, bundle: TrainingBundle) -> None:
        self.training_bundle = bundle
        lines = [f"Evaluated on {bundle.test_size:,} held-out test records ({bundle.target_name}):"]
        for r in bundle.results:
            if r.model_name in ("Linear Regression", "Random Forest"):
                lines.append(f"• {r.model_name}:  MAE = {r.mae:.2f}  |  RMSE = {r.rmse:.2f}  |  R2 = {r.r2:.3f}")
        self.ml_metrics_lbl.configure(text="\n".join(lines))

        best = bundle.results[0]
        y_true = bundle.y_test
        y_pred = bundle.predictions_by_model[best.model_name]

        def _draw_scatter(fig: Figure) -> None:
            ax = fig.add_subplot(111)
            if len(y_true) > 2000:
                idx = np.linspace(0, len(y_true) - 1, 2000, dtype=int)
                yt, yp = y_true[idx], y_pred[idx]
            else:
                yt, yp = y_true, y_pred
            ax.scatter(yt, yp, alpha=0.35, s=12, color="#0284C7")
            lim = max(float(np.max(yt)), float(np.max(yp)), 100.0)
            ax.plot([0, lim], [0, lim], "r--", linewidth=1.4, label="Ideal (Actual = Predicted)")
            ax.set_xlabel("Actual AQI", fontsize=8.5)
            ax.set_ylabel("Predicted AQI", fontsize=8.5)
            ax.set_title(f"{best.model_name} Accuracy (R2 = {best.r2:.3f}, MAE = {best.mae:.1f})", fontsize=9.5, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)

        self.ml_scatter_chart.draw(_draw_scatter)
        self._on_simple_predict()

    def _on_simple_predict(self) -> None:
        if self.training_bundle is None:
            return
        b = self.training_bundle
        station = self.pred_station_var.get()
        month = int(self.pred_month_var.get().split("-")[0].strip())
        val_in = float(self.pred_prev_var.get())
        overrides = {"AQI_lag1": val_in, "AQI_lag2": val_in * 0.96, "AQI_roll7_mean": val_in * 0.92} if b.mode == "next_day_aqi" else {"NO2": val_in}
        pred = predict_single_scenario(b, b.results[0].model_name, station, month, overrides)
        cat = get_aqi_category(pred)
        self.simple_pred_badge.configure(
            text=f"Estimated AQI in {station} (Month {month}):  {pred:.1f}  —  {cat.label}\nAdvice: {cat.advisory}",
            bg=cat.color,
            fg=cat.text_color,
        )
