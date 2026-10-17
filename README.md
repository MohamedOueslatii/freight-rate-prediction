# Freight Rate Prediction

Predicts the rate (USD) of freight loads. Trained on Jan-Oct 2025 loads (`data/train_test.csv`), it predicts the 12,000 loads of Nov-Dec 2025 in `data/validation.csv`.

## Approach in short
- **Target:** `log(rate / lane distance)`, converted back to dollars.
- **Model (hybrid):** a Huber-linear layer (time trend, log market index, log weight, daily quote level) extrapolates into unseen months; gradient-boosted trees (scikit-learn, L1 loss) learn lane, equipment, distance and quote effects from the residual.
- **Validation:** time-ordered (forward-chaining) folds: train on the past, test on the next two months. No random split (market_index is mostly a daily factor, so random folds leak day identity) and no month / day-of-year features (Nov-Dec never appears in training).
- **Results (mean % error on clean rows):** May-Jun 3.0%, Jul-Aug 5.2%, Sep-Oct 2.2%. The final metric is computed by Spotter on labels not available here.

## Data-quality handling
| Issue | Fix |
|---|---|
| ~1.4% corrupted `posted_rate` (rate/mile outside 1.2-4.0) | dropped from training only |
| ~0.6% negative `weight` | absolute value |
| ~0.6% missing `weight` | median in the linear layer; trees handle NaN natively |
| ~0.8% missing `market_index` | mean of the same day |
| noisy `distance` | lane median distance |
| December chart rows lack coordinates, market_index, quote_signal | coordinates from city lookup; daily means from `validation.csv` (no labels used) |

## Project structure
```
src/features.py   cleaning and feature engineering (imported, not run directly)
src/model.py      hybrid model (imported, not run directly)
src/explore.py    data-exploration charts -> eda/
src/cv.py         time-ordered validation -> cv_results.csv
src/compare.py    hybrid vs simpler models on the same folds -> model_comparison.csv
src/train.py      trains the final model, writes validation_predictions.csv and fills the December file
src/check.py      sanity checks on validation_predictions.csv
score.py          provided scorer (format check and December chart)
```

## Setup
```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```
Put the four provided CSV files in `data/` (they are not committed to this repository):
`train_test.csv`, `validation.csv`, `validation_predictions_template.csv`, `december_chart_inputs.csv`.

## Run (from the repository root)
```bash
python src/explore.py   # optional: exploration charts
python src/cv.py        # time-ordered validation
python src/compare.py   # optional: model comparison
python src/train.py     # final model -> validation_predictions.csv, data/december_chart_inputs.csv
python src/check.py     # optional: sanity checks
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
`train.py` overwrites `data/december_chart_inputs.csv` with the predicted rates. `score.py` writes `scorer_results/candidate_december.png`.