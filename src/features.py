"""Cleaning + feature engineering. Everything here is label-free, so it is safe to
fit on train+validation features together (e.g. lane median distance, daily market index)."""
import numpy as np
import pandas as pd

LIN = ["t", "lmi", "lw", "qs_day", "qs_dev"]            # linear (extrapolating) layer
GB = ["ld", "eq", "w", "mi", "quote_signal", "dow",      # boosted-tree layer (no time index!)
      "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "pickup_c", "delivery_c"]
EQUIP = ["Dry Van", "Flatbed", "Reefer"]
T0 = pd.Timestamp("2025-01-01")
# Label-outlier band on rate-per-mile. The train rpm histogram has a clear gap:
# ~98.6% of loads lie in [1.2, 4.0]; the rest are x0.2-0.5 or x3-6 corrupted rates.
RPM_LO, RPM_HI = 1.2, 4.0
DIST_OUTLIER = 0.15   # a distance more than 15% away from its lane median is treated as an error


def load(path):
    return pd.read_csv(path, parse_dates=["date"])


def build(frames, cities=None):
    """frames: dict name -> DataFrame (same schema, label optional). Returns one frame with a 'src' col."""
    al = pd.concat([f.assign(src=k) for k, f in frames.items()], ignore_index=True)
    al["lane"] = al.pickup + ">" + al.delivery
    # A load's own distance carries real price information (rate follows it, slope ~0.86 in the data),
    # so keep it. Only replace it by the lane median when it is clearly wrong (>15% off the lane median).
    # NB: the column keeps the name "lane_dist" for compatibility with the other scripts.
    lane_med = al.groupby("lane").distance.transform("median")
    dev = (al.distance / lane_med - 1).abs()
    al["lane_dist"] = np.where(~(dev <= DIST_OUTLIER), lane_med, al.distance)
    al["ld"] = np.log(al.lane_dist)
    # weight: sign-flipped in ~0.6% of rows -> abs(); NaN -> median (tree handles it, linear layer gets median)
    al["w"] = al.weight.abs()
    # market_index is a daily factor (within-day std ~0.025): impute missing from the day's mean
    day_mi = al.groupby("date").market_index.mean()
    al["mi"] = al.market_index.fillna(al.date.map(day_mi))
    al["lmi"] = np.log(al.mi)
    al["lw"] = np.log(al.w.fillna(al.w.median()))
    # quote_signal: slow daily level (day-to-day autocorr .93) + per-load deviation
    al["qs_day"] = al.groupby("date").quote_signal.transform("mean")
    al["qs_dev"] = al.quote_signal - al.qs_day
    al["t"] = (al.date - T0).dt.days.astype(float)
    al["dow"] = al.date.dt.dayofweek
    al["eq"] = pd.Categorical(al.equipment, categories=EQUIP).codes
    cities = cities if cities is not None else sorted(set(al.pickup) | set(al.delivery))
    for c in ("pickup", "delivery"):
        al[c + "_c"] = pd.Categorical(al[c], categories=cities).codes
    return al