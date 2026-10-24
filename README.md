# Mobile Game A/B Test: Moving the Progress Gate (Cookie Cats)

End-to-end A/B test analysis of 90,189 players from the Kaggle "Mobile Games A/B Testing - Cookie Cats" dataset.
The game tested moving its first progress gate from level 30 (control) to level 40 (treatment).

## Result
**Keep the gate at level 30.** Moving it to level 40 lowered 7-day retention from 19.02% to 18.20%
(-0.82 pp, 95% CI -1.33 to -0.31 pp, p = 0.0016, Holm-adjusted p = 0.0047). Bootstrap and Bayesian
analyses both put the probability that the control is better at 99.9%. 1-day retention and game rounds
showed no significant difference.

## What the analysis covers
| Step | Method |
|---|---|
| Data validation | Nulls, duplicates, bot/outlier detection (one player with ~50k rounds) |
| Sample ratio mismatch | Chi-square goodness of fit on the 50/50 split |
| Sensitivity | Minimum detectable effect at 80% power, required sample size |
| Primary test | Two-proportion z-test with 95% CI on the difference |
| Robustness | Bootstrap (5,000 resamples), Bayesian Beta-Binomial with expected loss |
| Engagement | Mann-Whitney U, Welch t-test, bootstrap CI on mean and median |
| Segments | 7-day retention by engagement bucket (descriptive, post-treatment) |
| Multiple testing | Holm step-down correction across 3 tests |
| Peeking | 1,000-run A/A simulation: daily peeking raises false positives from 4.2% to 21.5% |

## Files
- `ab_test_cookie_cats.ipynb` - executed notebook with all code, outputs and charts
- `build_notebook.py` - source that generates the notebook
- `results.json` - every reported number
- `charts/` - six PNG charts
- `data/cookie_cats.csv` - the dataset

## Run it
```bash
pip install pandas numpy scipy statsmodels matplotlib nbformat nbconvert ipykernel
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace ab_test_cookie_cats.ipynb
```

## Caveats
- The SRM check flags a small imbalance (44,700 vs 45,489, p = 0.009). In production this would be investigated before acting.
- No pre-experiment covariates, so CUPED variance reduction and balance checks are not possible on this dataset.
- Gate purchase revenue is not in the data; a full decision should weigh it alongside retention.
