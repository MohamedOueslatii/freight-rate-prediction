"""Fit on ALL labeled data (label outliers removed) and write validation + December predictions."""
import numpy as np, pandas as pd
from features import load, build, RPM_LO, RPM_HI
from model import HybridRateModel

tr = load("data/train_test.csv"); va = load("data/validation.csv")
dec = pd.read_csv("data/december_chart_inputs.csv", parse_dates=["date"])
tmpl = pd.read_csv("data/validation_predictions_template.csv")

# --- December chart rows lack coordinates / market_index / quote_signal: fill them label-free
coords = pd.concat([tr[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates("pickup"),
                    ]).set_index("pickup")
dcoords = tr[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates("delivery").set_index("delivery")
dec["pickup_lat"] = dec.pickup.map(coords.pickup_lat); dec["pickup_lon"] = dec.pickup.map(coords.pickup_lon)
dec["delivery_lat"] = dec.delivery.map(dcoords.delivery_lat); dec["delivery_lon"] = dec.delivery.map(dcoords.delivery_lon)
vd = va.groupby("date").agg(market_index=("market_index", "mean"), quote_signal=("quote_signal", "mean"))
dec["market_index"] = dec.date.map(vd.market_index); dec["quote_signal"] = dec.date.map(vd.quote_signal)
dec["load_id"] = [f"DEC-{i:03d}" for i in range(len(dec))]
assert dec[["pickup_lat", "delivery_lat", "market_index", "quote_signal"]].notna().all().all()

al = build({"tr": tr, "va": va, "dec": dec.drop(columns="predicted_rate")})
# the December lane distance is given (360 mi); keep the given value rather than lane median if lane unseen
lane_seen = al[al.src != "dec"].lane.unique()
m = (al.src == "dec") & ~al.lane.isin(lane_seen)
al.loc[m, "lane_dist"] = al.loc[m, "distance"]; al["ld"] = np.log(al.lane_dist)

t = al[al.src == "tr"].copy(); t["y"] = np.log(t.posted_rate / t.lane_dist)
t = t[t.y.between(np.log(RPM_LO), np.log(RPM_HI))]
print(f"training rows after label-outlier removal: {len(t)} / {len(tr)}")
model = HybridRateModel().fit(t, t.y)

v = al[al.src == "va"]
out = pd.DataFrame({"load_id": v.load_id.values, "predicted_rate": model.predict_rate(v).round(2)})
assert list(out.load_id) == list(tmpl.load_id) and out.predicted_rate.notna().all() and (out.predicted_rate > 0).all()
out.to_csv("validation_predictions.csv", index=False)

d = al[al.src == "dec"]; dec["predicted_rate"] = model.predict_rate(d).round(2)
dec[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]].assign(
    date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_csv("data/december_chart_inputs.csv", index=False)
print(out.predicted_rate.describe().round(1)); print(dec[["date", "market_index", "predicted_rate"]].head(10).to_string())
