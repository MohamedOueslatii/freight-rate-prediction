"""Exploration: prints key data-quality counts and saves 4 charts to eda/ (for the report / video)."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import load, RPM_LO, RPM_HI

os.makedirs("eda", exist_ok=True)
tr = load("data/train_test.csv"); va = load("data/validation.csv")
tr["rpm"] = tr.posted_rate / tr.distance
C = "#064A56"

# ---------- 1. Data-quality summary
print("=== DATA OVERVIEW")
print(f"train rows: {len(tr):,}  ({tr.date.min():%Y-%m-%d} -> {tr.date.max():%Y-%m-%d})")
print(f"valid rows: {len(va):,}  ({va.date.min():%Y-%m-%d} -> {va.date.max():%Y-%m-%d})  <- future, no labels")
print("\n=== DATA QUALITY (train)")
out = (~tr.rpm.between(RPM_LO, RPM_HI)).sum()
print(f"corrupted rates (rate/mile outside {RPM_LO}-{RPM_HI}): {out} ({out/len(tr):.2%})")
print(f"negative weights: {(tr.weight < 0).sum()} ({(tr.weight < 0).mean():.2%})")
print(f"missing weight: {tr.weight.isna().sum()} ({tr.weight.isna().mean():.2%})")
print(f"missing market_index: {tr.market_index.isna().sum()} ({tr.market_index.isna().mean():.2%})")
lane = tr.pickup + ">" + tr.delivery
dev = (tr.distance / tr.groupby(lane).distance.transform("median"))
print(f"distance >15% away from its lane median: {(abs(dev - 1) > 0.15).sum()}")
print("\nmissing values in validation:"); print(va.isna().sum()[lambda s: s > 0].to_string())

clean = tr[tr.rpm.between(RPM_LO, RPM_HI)].copy()
daily = clean.groupby("date").agg(rpm=("rpm", "median"), mi=("market_index", "mean"))

# ---------- Chart 1: rate histogram with outlier band
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(np.log10(tr.rpm), bins=120, color=C)
for v in (RPM_LO, RPM_HI): ax.axvline(np.log10(v), color="red", ls="--")
ax.set_xlabel("log10(rate per mile)"); ax.set_ylabel("loads")
ax.set_title("Rate per mile: ~1.4% corrupted values outside the red band")
fig.tight_layout(); fig.savefig("eda/1_rate_outliers.png", dpi=150); plt.close(fig)

# ---------- Chart 2: trend over time (train) + validation market index
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(daily.index, daily.rpm.rolling(14, min_periods=1).mean(), color=C, label="median rate/mile (14d avg)")
ax.set_ylabel("rate per mile ($)")
ax2 = ax.twinx()
dm = pd.concat([tr, va]).groupby("date").market_index.mean()
ax2.plot(dm.index, dm.rolling(7, min_periods=1).mean(), color="orange", alpha=.8, label="market_index (7d avg)")
ax2.axvline(va.date.min(), color="grey", ls="--"); ax2.set_ylabel("market_index")
ax.set_title("Rates (teal) vs market_index (orange): slow trends differ; dashed = start of validation", fontsize=10)
fig.tight_layout(); fig.savefig("eda/2_trend_and_market.png", dpi=150); plt.close(fig)

# ---------- Chart 3: weekly cycle of market index
fig, ax = plt.subplots(figsize=(9, 3.5))
sub = dm.loc["2025-10-01":"2025-12-31"]
ax.plot(sub.index, sub.values, color=C); ax.set_title("market_index has a strong 7-day cycle (Oct-Dec)")
fig.autofmt_xdate(); fig.tight_layout(); fig.savefig("eda/3_weekly_cycle.png", dpi=150); plt.close(fig)

# ---------- Chart 4: rate per mile vs distance, by equipment
fig, ax = plt.subplots(figsize=(9, 4))
clean["dist_bin"] = pd.cut(clean.distance, [0, 150, 300, 500, 800, 1200, 2000, 4000])
for eq, g in clean.groupby("equipment"):
    s = g.groupby("dist_bin", observed=True).rpm.median(); ax.plot(range(len(s)), s.values, marker="o", label=eq)
ax.set_xticks(range(7)); ax.set_xticklabels(["<150", "150-300", "300-500", "500-800", "800-1200", "1200-2000", ">2000"])
ax.set_xlabel("distance (miles)"); ax.set_ylabel("median rate per mile ($)"); ax.legend()
ax.set_title("Short hauls cost more per mile; equipment shifts the level")
fig.tight_layout(); fig.savefig("eda/4_distance_equipment.png", dpi=150); plt.close(fig)
print("\nCharts saved in eda/")