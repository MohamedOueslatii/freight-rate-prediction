"""Sanity checks on the final outputs. Run after train.py."""
import numpy as np, pandas as pd

pred = pd.read_csv("validation_predictions.csv")
tmpl = pd.read_csv("data/validation_predictions_template.csv")
va = pd.read_csv("data/validation.csv", parse_dates=["date"])
tr = pd.read_csv("data/train_test.csv")

ok = lambda c, msg: print(("PASS  " if c else "FAIL  ") + msg)
ok(list(pred.columns) == ["load_id", "predicted_rate"], "columns are exactly load_id,predicted_rate")
ok(len(pred) == 12000, f"12,000 rows (got {len(pred)})")
ok(pred.load_id.is_unique, "no duplicate load_id")
ok(list(pred.load_id) == list(tmpl.load_id), "same load_id values, same order as the template")
ok(pred.predicted_rate.notna().all() and (pred.predicted_rate > 0).all(), "no missing or non-positive predictions")

m = va.merge(pred, on="load_id"); m["rpm"] = m.predicted_rate / m.distance
tr_rpm = (tr.posted_rate / tr.distance).median()
print(f"\nmedian predicted rate/mile: {m.rpm.median():.2f}  (train median: {tr_rpm:.2f})")
ok(0.8 * tr_rpm < m.rpm.median() < 1.2 * tr_rpm, "predicted level is in the same range as training")
print("\nmedian predicted rate/mile by equipment (should differ, like in training):")
print(m.groupby("equipment").rpm.median().round(2).to_string())
print("\nmean predicted rate by month:"); print(m.groupby(m.date.dt.to_period("M")).predicted_rate.mean().round(1).to_string())
c = np.corrcoef(m.distance, m.predicted_rate)[0, 1]
ok(c > 0.8, f"longer trips cost more (corr distance vs rate = {c:.2f})")