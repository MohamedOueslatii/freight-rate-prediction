"""Compares the final hybrid model with simpler alternatives on the same time-ordered folds.
Produces the model-comparison table used in the report (mean % error on clean test rows)."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as H
from sklearn.linear_model import HuberRegressor
from features import load, build, LIN, GB, RPM_LO, RPM_HI
from model import HybridRateModel

tr = load("data/train_test.csv"); va = load("data/validation.csv")
al = build({"tr": tr, "va": va}); t = al[al.src == "tr"].copy()
t["y"] = np.log(t.posted_rate / t.lane_dist); t["rpm"] = np.exp(t.y)
clean = t.y.between(np.log(RPM_LO), np.log(RPM_HI))
folds = [("2025-05-01", "2025-06-30"), ("2025-07-01", "2025-08-31"), ("2025-09-01", "2025-10-31")]

def same_price(trn, te):  return np.full(len(te), trn.rpm.median()) * te.lane_dist.values
def by_equipment(trn, te): return te.equipment.map(trn.groupby("equipment").rpm.median()).values * te.lane_dist.values
def linear_only(trn, te):
    mu = trn[LIN].values.mean(0)
    m = HuberRegressor(max_iter=1000).fit(trn[LIN].values - mu, trn.y)
    return np.exp(m.predict(te[LIN].values - mu)) * te.lane_dist.values
def trees_only(trn, te):
    f = GB + ["t"]; ci = [f.index(c) for c in ("eq", "pickup_c", "delivery_c")]
    m = H(loss="absolute_error", max_iter=400, learning_rate=0.05, min_samples_leaf=30,
          categorical_features=ci, early_stopping=False).fit(trn[f], trn.y)
    return np.exp(m.predict(te[f])) * te.lane_dist.values
def hybrid(trn, te): return HybridRateModel().fit(trn, trn.y).predict_rate(te)

models = {"Same price per mile for everyone": same_price, "Price per mile by equipment": by_equipment,
          "Linear layer only": linear_only, "Trees only (with day count)": trees_only, "Hybrid (chosen)": hybrid}
rows = []
for name, fn in models.items():
    errs = []
    for a, b in folds:
        trn = t[(t.date < a) & clean]; te = t[(t.date >= a) & (t.date <= b) & clean]
        p = fn(trn, te); y = te.posted_rate.values; errs.append(np.mean(np.abs(p - y) / y))
    rows.append([name] + [f"{e:.1%}" for e in errs] + [f"{np.mean(errs):.1%}"])
out = pd.DataFrame(rows, columns=["Model (mean % error, clean rows)", "May-Jun", "Jul-Aug", "Sep-Oct", "Average"])
print(out.to_string(index=False)); out.to_csv("model_comparison.csv", index=False)