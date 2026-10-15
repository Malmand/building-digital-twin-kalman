"""Figures from results/ (run after the steps). Each figure is skipped if its results are missing."""
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotstyle import C, panel, save, subplots  # noqa: I001  (plotstyle sets the Agg backend)
from twin import data as D

RES = Path(__file__).parent / "results"
FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)
MC = {"M1": C["sky"], "M2": C["orange"], "M3": C["blue"]}
EXAMPLE_ZONE = "F4z2"


def fig1_data():
    """Data availability by zone and month, and one week of one zone."""
    t = D.raw_table()
    zones = D.zones()
    avail = pd.DataFrame({z: t[f"Floor{z[1]}_z{z[3]}_S1(degC)"].notna() for z in zones}).groupby(
        t.index.to_period("M")).mean()
    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1.35], hspace=0.55, wspace=0.75)
    ax = fig.add_subplot(gs[:, 0])
    im = ax.imshow(avail.T.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(zones)), zones, fontsize=6)
    lab = [p.strftime("%b %y") if p.month in (7, 10, 1, 4) else "" for p in avail.index]
    ax.set_xticks(range(len(avail)), lab, fontsize=6, rotation=90)
    ax.set_xlabel("Month")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("Temperature data available (-)", fontsize=7)
    panel(ax, "A", x=-0.3, y=1.01)
    df = D.zone_frame(EXAMPLE_ZONE, "2019-06-17", "2019-06-23 23:55")
    a1 = fig.add_subplot(gs[0, 1])
    a1.plot(df.index, df["Ti"], color=C["blue"], lw=0.8, label=f"Zone air ({EXAMPLE_ZONE})")
    a1.plot(df.index, df["T_out"], color=C["vermillion"], lw=0.8, ls="--", label="Outdoor (ERA5)")
    a1.set_ylabel("Temperature (°C)")
    a1.legend(frameon=False, fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.18))
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(df.index, df["P_ac"], color=C["black"], lw=0.7, label="Air-conditioning")
    a2.plot(df.index, df["P_int"], color=C["green"], lw=0.7, ls=":", label="Lighting and plugs")
    a2.set_ylabel("Electric power (kW)")
    a2.legend(frameon=False, fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.18))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%a"))
    a1.tick_params(labelbottom=False)
    panel(a1, "B", x=-0.16, y=1.12)
    panel(a2, "C", x=-0.16, y=1.12)
    fig.savefig(FIG / "fig1_data.png")
    plt.close(fig)


def fig2_identification():
    """Forecast error against horizon for the three levels of detail, and one test week."""
    from twin.runner import load_model, run_twin
    d = pd.read_csv(RES / "identification.csv")
    d = d[d["model"].isin(["M1", "M2", "M3"])]
    hz = {"rmse_5min": 5 / 60, "rmse_1h": 1, "rmse_3h": 3, "rmse_6h": 6}
    fig = plt.figure(figsize=(7.4, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.7, 1.6], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    for key, mk, ls in (("M1", "o", ":"), ("M2", "s", "--"), ("M3", "^", "-")):
        g = d[d["model"] == key]
        med = [g[c].median() for c in hz]
        lo = [g[c].quantile(0.25) for c in hz]
        hi = [g[c].quantile(0.75) for c in hz]
        x = list(hz.values())
        ax.fill_between(x, lo, hi, color=MC[key], alpha=0.18, lw=0)
        ax.plot(x, med, marker=mk, ls=ls, color=MC[key], lw=1, ms=3.5, label=f"{SPEC_NAMES[key]}")
    ax.set(xlabel="Forecast horizon (h)", ylabel="Temperature error, RMSE (K)", ylim=(0, None))
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    panel(ax, "A", x=-0.3)
    a2 = fig.add_subplot(gs[1])
    vals = [d[d["model"] == k]["acf_max_lag1_12"].to_numpy() for k in ("M1", "M2", "M3")]
    bp = a2.boxplot(vals, widths=0.55, showfliers=False, medianprops={"color": "k"})
    for i, v in enumerate(vals):
        a2.plot(np.full(len(v), i + 1) + np.linspace(-0.12, 0.12, len(v)), v, "o", ms=2, color=MC[f"M{i+1}"], alpha=0.8)
    a2.set_xticks([1, 2, 3], ["M1", "M2", "M3"])
    a2.set_ylabel("Largest residual autocorrelation,\nlags 5-60 min (-)")
    a2.axhline(2 / np.sqrt(4000), color="k", lw=0.5, ls=":")
    panel(a2, "B", x=-0.55)
    a3 = fig.add_subplot(gs[2])
    m = load_model(EXAMPLE_ZONE, "M3")
    df = D.zone_frame(EXAMPLE_ZONE, "2019-06-17", "2019-06-30 23:55")
    o = run_twin(m, df, track=False, horizons=(72,), stride=1).loc["2019-06-24":"2019-06-26 23:55"]
    a3.fill_between(o.index, o["fc72"] - 1.96 * o["fc72_std"], o["fc72"] + 1.96 * o["fc72_std"],
                    color=C["blue"], alpha=0.2, lw=0, label="95 % band")
    a3.plot(o.index, o["fc72"], color=C["blue"], lw=0.9, label="Forecast made 6 h earlier (M3)")
    a3.plot(o.index, o["Ti"], color="k", lw=0.7, ls="--", label="Measured")
    a3.set_ylabel(f"Air temperature, {EXAMPLE_ZONE} (°C)")
    a3.xaxis.set_major_locator(mdates.DayLocator())
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    a3.set_ylim(18.5, 30.5)
    a3.legend(frameon=False, fontsize=6, loc="upper right", ncol=1)
    panel(a3, "C", x=-0.17)
    fig.savefig(FIG / "fig2_identification.png")
    plt.close(fig)


def fig3_operation():
    """Six months of operation: forecast error by month, calibration of the uncertainty, and the
    tracked cooling effectiveness."""
    f = RES / "twin_monthly.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    daily = pd.read_csv(RES / "twin_daily.csv", parse_dates=[0], index_col=0)
    fig, ax = subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1.2, 0.8, 1.2]})
    months = sorted(d["month"].unique())
    x = np.arange(len(months))
    for (ver, h), col, mk, ls in ((("fixed", "6h"), C["blue"], "o", "-"), (("tracking", "6h"), C["orange"], "s", "--"),
                                   (("fixed", "1h"), C["blue"], "o", ":"), (("tracking", "1h"), C["orange"], "s", ":")):
        g = d[(d["version"] == ver) & (d["horizon"] == h)].groupby("month")["rmse"]
        med = g.median().reindex(months)
        ax[0].plot(x, med, marker=mk, ls=ls, color=col, lw=1, ms=3, label=f"{h} ahead, {ver}")
    ax[0].set_xticks(x, [pd.Period(m).strftime("%b") for m in months])
    ax[0].set(ylabel="Median RMSE over zones (K)", ylim=(0, None), xlabel="Month (2019)")
    ax[0].legend(frameon=False, fontsize=6, loc="lower right")
    panel(ax[0], "A", x=-0.28)
    cov = d.groupby(["version", "horizon", "zone"])["coverage95"].mean().reset_index()
    for i, (ver, h) in enumerate((("fixed", "1h"), ("fixed", "6h"), ("tracking", "1h"), ("tracking", "6h"))):
        v = cov[(cov["version"] == ver) & (cov["horizon"] == h)]["coverage95"].to_numpy() * 100
        ax[1].plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, "o", ms=2.2,
                   color=C["blue"] if ver == "fixed" else C["orange"], alpha=0.8)
        ax[1].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=1)
    ax[1].axhline(95, color="k", lw=0.6, ls=":")
    ax[1].set_xticks(range(4), ["1 h\nfixed", "6 h\nfixed", "1 h\ntrack", "6 h\ntrack"], fontsize=6.5)
    ax[1].set_ylabel("Measurements inside the\n95 % forecast band (%)")
    panel(ax[1], "B", x=-0.45)
    for z, g in daily.groupby("zone"):
        ax[2].plot(g.index, g["kappa"], lw=0.5, color=C["sky"], alpha=0.6)
    med = daily.groupby(level=0)["kappa"].median()
    ax[2].plot(med.index, med, lw=1.3, color=C["blue"], label="Median over zones")
    ax[2].axhline(0.8, color=C["vermillion"], lw=0.8, ls="--", label="Alarm limit")
    ax[2].set(ylabel="Tracked cooling effectiveness κ (-)")
    ax[2].xaxis.set_major_locator(mdates.MonthLocator())
    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax[2].legend(frameon=False, fontsize=6, loc="lower left")
    panel(ax[2], "C", x=-0.38)
    save(fig, FIG / "fig3_operation.png")


def fig4_hybrid():
    f = RES / "hybrid.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    methods = ["persistence", "ML only", "twin", "hybrid"]
    cols = {"persistence": C["black"], "ML only": C["green"], "twin": C["blue"], "hybrid": C["orange"]}
    fig, ax = subplots(1, 2, figsize=(6.4, 2.6), sharey=False)
    for a, h, lab in ((ax[0], "1h", "A"), (ax[1], "6h", "B")):
        g = d[d["horizon"] == h]
        for i, m in enumerate(methods):
            v = g[g["method"] == m]["rmse"].to_numpy()
            a.plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, "o", ms=2.2, color=cols[m], alpha=0.8)
            a.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=1)
        a.set_xticks(range(len(methods)), ["Persist-\nence", "ML\nonly", "Twin", "Twin +\nML"], fontsize=7)
        a.set(ylabel=f"RMSE {h} ahead (K)", ylim=(0, None))
        panel(a, lab, x=-0.22)
    save(fig, FIG / "fig4_hybrid.png")


def fig5_diagnosis():
    f = RES / "diagnosis_injected.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fa = pd.read_csv(RES / "diagnosis_false_alarms.csv")
    order = ["bias 1 K", "bias 2 K", "drift", "frozen", "cooling"]
    names = {"bias 1 K": "Offset\n1 K", "bias 2 K": "Offset\n2 K", "drift": "Drift\n0.5 K/d",
             "frozen": "Frozen", "cooling": "Cooling\n-30 %"}
    fig, ax = subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    m = pd.read_csv(RES / "twin_monthly.csv")
    acc = m[(m["version"] == "fixed") & (m["horizon"] == "6h")].groupby("zone")["rmse"].median()
    d["good"] = d["zone"].map(acc) < 0.6
    xs = np.arange(len(order))
    for off, good, col, lab in ((-0.2, True, C["blue"], "Zones with 6-h error < 0.6 K"),
                                (0.2, False, C["sky"], "Zones with 6-h error ≥ 0.6 K")):
        rate = d[d["good"] == good].groupby("fault")["detected"].mean().reindex(order) * 100
        n = d[d["good"] == good]["zone"].nunique()
        ax[0].bar(xs + off, rate, 0.4, color=col, edgecolor="k", lw=0.4, label=f"{lab} ({n})")
    ax[0].set_xticks(xs, [names[o] for o in order], fontsize=6.5)
    ax[0].set(ylabel="Injected faults detected (%)", ylim=(0, 125))
    ax[0].set_yticks([0, 25, 50, 75, 100])
    ax[0].legend(frameon=False, fontsize=5.5, loc="upper left")
    panel(ax[0], "A", x=-0.25)
    for i, o in enumerate(order):
        v = d[(d["fault"] == o) & d["detected"]]["delay_h"].to_numpy()
        if len(v):
            ax[1].plot(np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), v, "o", ms=2, color=C["orange"], alpha=0.7)
            ax[1].plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="k", lw=1)
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(len(order)), [names[o] for o in order], fontsize=6.5)
    ax[1].set_ylabel("Time to alarm (h)")
    panel(ax[1], "B", x=-0.25)
    cols = ["cusum_alarms_per_30d", "frozen_alarms_per_30d", "cooling_alarms_per_30d"]
    for i, c in enumerate(cols):
        v = fa[c].dropna().to_numpy()
        ax[2].plot(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, "o", ms=2.2, color=C["vermillion"], alpha=0.8)
        ax[2].plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=1)
    ax[2].set_xticks(range(3), ["Offset/\ndrift", "Frozen", "Cooling"], fontsize=6.5)
    ax[2].set_ylabel("Alarms per 30 days\non recorded data (-)")
    panel(ax[2], "C", x=-0.45)
    save(fig, FIG / "fig5_diagnosis.png")


def fig6_flexibility():
    f = RES / "flex_validation.csv"
    if not f.exists():
        return
    v = pd.read_csv(f)
    s = pd.read_csv(RES / "flex_scenarios.csv")
    fig, ax = subplots(1, 3, figsize=(7.4, 2.7))
    for lab, col, mk in (("30min", C["sky"], "o"), ("2h", C["blue"], "^")):
        g = v[v["after"] == lab]
        ax[0].plot(g["measured_rise"], g["predicted_rise"], mk, ms=1.8, color=col, alpha=0.5,
                   label=f"{lab.replace('min', ' min').replace('2h', '2 h')} after the cut")
    lim = [min(v["measured_rise"].min(), v["predicted_rise"].min()), max(v["measured_rise"].max(), v["predicted_rise"].max())]
    ax[0].plot(lim, lim, color="k", lw=0.6, ls=":")
    ax[0].set(xlabel="Measured rise (K)", ylabel="Rise predicted by the twin (K)")
    ax[0].legend(frameon=False, fontsize=6, loc="upper left", markerscale=2.5)
    panel(ax[0], "A", x=-0.28)
    order = ["off 30 min", "off 60 min", "off 120 min", "half power 120 min"]
    lbl = ["Off\n30 min", "Off\n60 min", "Off\n120 min", "Half power\n120 min"]
    for i, sc in enumerate(order):
        g = s[s["scenario"] == sc]
        ax[1].boxplot([g["max_extra_rise"]], positions=[i], widths=0.5, showfliers=False, medianprops={"color": "k"})
    ax[1].set_xticks(range(4), lbl, fontsize=6.5)
    ax[1].set_ylabel("Extra temperature rise (K)")
    panel(ax[1], "B", x=-0.28)
    agg = s.groupby(["day", "scenario"]).agg(shed=("shed_kWh", "sum")).reset_index()
    # zone-events pushed above 26 C with probability > 50 % that would not have been without the cut
    risk = s.assign(exceed=(s["p_exceed_max"] > 0.5) & (s["base_p_exceed_max"] <= 0.5)).groupby("scenario")["exceed"].mean() * 100
    shed = agg.groupby("scenario")["shed"].median()
    ax[2].bar(range(4), [shed[o] for o in order], color=C["green"], edgecolor="k", lw=0.4)
    ax[2].set_xticks(range(4), lbl, fontsize=6.5)
    ax[2].set_ylabel("Electricity shed, whole building\n(kWh per event, median)")
    a2 = ax[2].twinx()
    a2.plot(range(4), [risk[o] for o in order], "D", color=C["vermillion"], ms=4)
    a2.set_ylabel("Zone-events pushed above 26 °C (%)", color=C["vermillion"])
    a2.spines["right"].set_visible(True)
    a2.tick_params(axis="y", colors=C["vermillion"])
    panel(ax[2], "C", x=-0.42)
    save(fig, FIG / "fig6_flexibility.png")


SPEC_NAMES = {"M1": "M1, 1R1C", "M2": "M2, 2R2C", "M3": "M3, 4R3C"}


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6"]
    for name, f in list(globals().items()):
        if name.startswith("fig") and name[3] in which and callable(f):
            f()
