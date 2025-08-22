ML Integration Notes

Goal
- Bring core ML practices from Stefan Jansen's "machine-learning-for-trading"
  into this repo: clean pipelines, walk-forward CV, reproducible features,
  model versioning, and backtest integration.

Short-term plan
1. Provide a small, runnable pipeline baseline: `research/ml_pipeline.py`.
2. Use walk-forward CV (TimeSeriesSplit) to produce multiple models and
   evaluate stability before backtesting.
3. Save model artifacts to `models/` and record parameters.
4. Add tests for feature generation and small smoke training run.

Next steps (medium-term)
- Replace ad-hoc features with a feature store module. Create a
  `src/ml/features.py` that centralizes indicator generation, scaling,
  and feature groups.
- Implement backtest integration: load the walk-forward models and produce
  signals for `src/backtesting/backtester.py` to evaluate economic
  performance (transaction costs, slippage, sizing).
- Add Optuna hyperparameter tuning with proper CV and pruning.
- Add light-weight examples for LightGBM and XGBoost, and a GPU-aware
  training script for deep learning models.

Quality gates
- Lint (black/isort) and type-check (mypy).
- Unit tests: feature generation and small training smoke test.
- CI: Add a GH action to run the smoke tests (fast, uses small subset of
  data).

References
- https://github.com/stefan-jansen/machine-learning-for-trading
- Relevant chapters: data engineering, model validation, walk-forward CV
