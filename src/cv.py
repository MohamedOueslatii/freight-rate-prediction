"""Forward-chaining (time-ordered) validation. Mimics the real task: fit on the past, score the future."""
import numpy as np, pandas as pd
from features import load, build, RPM_LO, RPM_HI
from model import HybridRateModel

tr = load("data/train_test.csv"); va = load("data/validation.csv")
al = build({"tr": tr, "va": va}); t = al[al.src == "tr"].copy()
t["y"] = np.log(t.posted_rate / t.lane_dist)
clean = t.y.between(np.log(RPM_LO), np.log(RPM_HI))
# last fold has a 2-month horizon like the real Nov-Dec test; earlier folds check stability
folds = [("2025-05-01", "2025-06-30"), ("2025-07-01", "2025-08-31"), ("2025-09-01", "2025-10-31")]
rows = []
for a, b in folds:
    trn = t[(t.date < a) & clean]; te = t[(t.date >= a) & (t.date <= b)]
    p = HybridRateModel().fit(trn, trn.y).predict_rate(te); y = te.posted_rate.values; ok = clean[te.index].values
    ape = np.abs(p - y) / y
    rows.append(dict(fold=f"{a[:7]}..{b[:7]}", n_train=len(trn), n_test=len(te), MAE_all=np.abs(p-y).mean(),
                     MAE_clean=np.abs(p-y)[ok].mean(), MAPE_clean=ape[ok].mean(), MedAPE=np.median(ape),
                     bias_logclean=np.mean(np.log(y/p)[ok])))
r = pd.DataFrame(rows).round(4); print(r.to_string(index=False)); r.to_csv("cv_results.csv", index=False)
