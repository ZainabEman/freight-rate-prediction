"""Narrative content for the technical report.

Every fact here is transcribed from an artifact written by Phases 1-8. No value
is recomputed and no conclusion is altered; this module only organises and
explains what was already measured.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Project timeline: (phase, title, goal, deliverables, outcome)
# --------------------------------------------------------------------------- #
PHASES: list[tuple[str, str, str, str, str]] = [
    ("1", "Data Audit",
     "Establish ground truth about an inherited codebase and its two datasets before writing "
     "anything.",
     "reports/data_audit.md, reports/feature_dictionary.md, schema and quality validators.",
     "Four material data faults identified, plus a silent code defect that was destroying every "
     "feature name."),
    ("2", "Exploratory Analysis",
     "Understand what actually drives a freight rate, and quantify it.",
     "reports/exploratory_data_analysis.md, reports/business_insights.md, 27 figures.",
     "Distance confirmed as the dominant driver (r = 0.909); directional and equipment premiums "
     "quantified."),
    ("3", "Foundation Repair and Feature Engineering",
     "Repair the inherited defects and build a production preprocessing pipeline.",
     "156-feature pipeline, integrity guards, temporal split utility, dual inference path, "
     "80 tests.",
     "Feature names preserved end-to-end and asserted; leakage prevented structurally."),
    ("4", "Baselines and Evaluation Framework",
     "Establish an honest reference bar and a time-aware evaluation harness.",
     "reports/baseline_models.md, seven baselines, evaluation harness, 87 tests.",
     "Best baseline MAE $145.24. The rate-per-mile heuristic alone reached R2 = 0.807."),
    ("5", "Advanced Models and Tuning",
     "Determine whether gradient boosting can beat a well-specified linear model.",
     "reports/model_comparison.md, five tuned model families, 93 tests.",
     "CatBoost selected at MAE $132.15 - but within $4 of two other families."),
    ("6", "Explainability and Error Analysis",
     "Explain what the model learned and characterise where it fails.",
     "Three importance views, 28 figures, residual diagnostics, business insights.",
     "Heavy tail identified as the dominant error structure; a stated claim about market context "
     "was corrected by the evidence."),
    ("7", "Final Training and Predictions",
     "Produce the submission with exactly one training run.",
     "validation_predictions.csv, December chart, models/final_model.joblib.",
     "Smearing correction cut MAE 13% to $114.99; the official scorer passed."),
    ("8", "Documentation and Packaging",
     "Make the work legible to someone who never opens the repository.",
     "README, this technical report, submission checklist, binary-safe git configuration.",
     "Complete submission package, verified end-to-end."),
]

# --------------------------------------------------------------------------- #
# Engineering decisions: (title, problem, alternatives, chosen, reason, evidence)
# --------------------------------------------------------------------------- #
DECISIONS: list[tuple[str, str, str, str, str, str]] = [
    ("D1. Temporal validation instead of a random split",
     "The labelled window ends 2025-10-31 and the scoring window begins 2025-11-01, with zero "
     "overlap. Any split that samples across time measures a task the model will never face.",
     "Random k-fold; a single random holdout; grouping by lane.",
     "Hold out the final contiguous block of the development window (2025-09-01 to 2025-10-31) and "
     "cross-validate with an expanding window over date-ordered rows.",
     "A random split scores interpolation between known dates. The real task is forward "
     "extrapolation into two unseen months, which is a strictly harder problem. Reporting the "
     "easier number would have been misleading rather than merely optimistic.",
     "Train and scoring windows are provably disjoint. Every CV fold was asserted to train only on "
     "rows earlier than the rows it scores."),

    ("D2. Holdout scored once, never tuned against",
     "A holdout consulted repeatedly during model selection silently becomes a training set, and "
     "the final estimate inflates.",
     "Select on cross-validation score alone; iterate freely against the holdout.",
     "Tune exclusively on cross-validation within the training block; score the holdout once per "
     "model.",
     "This preserves the holdout as an unbiased estimate of forward performance, which is the only "
     "number that predicts scoring-set behaviour.",
     "CV and holdout rankings genuinely disagreed - RandomForest scored CV $167.68 but holdout "
     "$144.85. A contaminated holdout would have destroyed exactly this signal."),

    ("D3. Leakage prevented structurally, not by convention",
     "Fitting a preprocessor once on all training data leaks future feature distributions into "
     "earlier CV folds. The failure is silent and invisible in the metrics.",
     "Fit the preprocessor once and reuse it (faster, but leaks); refit manually per fold (correct "
     "but easy to get wrong under refactoring).",
     "Make preprocessing a step inside the searched pipeline, so scikit-learn clones and refits it "
     "independently within every fold.",
     "Correctness becomes a property of the object graph rather than of the author's discipline. A "
     "later refactor cannot reintroduce the leak without deleting the structure itself.",
     "No imputation median, scaler statistic or one-hot vocabulary is ever learned from data later "
     "than the rows it scores. Verified by assertions on fold ordering and disjointness."),

    ("D4. Explicit leakage and alignment assertions",
     "The inherited pipeline had already demonstrated that silent failure is the real risk: a "
     "swallowed exception was renaming all 142 processed columns to integers on every run.",
     "Trust code review; rely on downstream metrics to reveal problems.",
     "Assert on every run that feature names survive, that train and inference columns match in "
     "content and order, that no NaN remains, and that neither the target nor load_id reaches the "
     "feature matrix.",
     "Metrics do not reveal misalignment - a positionally shuffled matrix still trains and still "
     "produces a number. Only an explicit check catches it.",
     "assert_feature_names_preserved() is a direct regression guard for the inherited defect. An "
     "independent refit is also asserted to reproduce the feature matrix exactly."),

    ("D5. Cyclical date encoding, and no monotone time feature",
     "date_year is constant at 2025 and carries zero information. Raw date_month takes values "
     "{11, 12} at scoring against {1..10} in training - entirely outside the learned support.",
     "Raw month and day integers (the inherited implementation); one-hot months; a linear time "
     "trend or day counter.",
     "Encode date as sine/cosine pairs of day-of-year and day-of-week. Explicitly disable the "
     "monotone days-since-reference feature.",
     "Tree models cannot extrapolate: given month 11 they fall into the month-10 leaf and every "
     "scoring date collapses to a single value. Cyclical encodings are bounded in [-1, 1] and "
     "continuous across the year boundary, so November and December land inside the training "
     "support by construction.",
     "This is directly visible in the delivered output: the December chart shows a genuine weekly "
     "rhythm. Under the inherited encoding it would have been a flat line."),

    ("D6. Negative weights repaired by sign flip, not discarded",
     "292 training rows carry a negative weight, which is physically impossible.",
     "Drop the rows; treat them as missing and impute; leave them untouched.",
     "Apply abs(), then clip to the observed physical envelope [5,000, 47,500].",
     "The distributional test distinguishes a data-entry fault from a distinct population. Because "
     "the magnitudes match the healthy population exactly, the sign is the only corrupted part and "
     "the magnitude is recoverable - dropping or imputing would discard real information.",
     "Mean of |negative| = 31,724 against a positive mean of 31,415, with both spanning an "
     "identical [5,000, 47,500] range."),

    ("D7. Missing-value indicators alongside imputation",
     "weight and market_index are missing in a small share of rows, but the rate roughly doubles "
     "between training and scoring.",
     "Impute silently; drop incomplete rows; model-based imputation.",
     "Emit weight_is_missing and market_index_is_missing before median imputation.",
     "Median imputation alone erases the fact that a value was absent. Since the missingness rate "
     "itself shifts between the two windows, that fact is signal about the scoring distribution, "
     "not noise to be smoothed away.",
     "weight missing 0.63% in train against 1.38% in validation; market_index 0.78% against 2.08%."),

    ("D8. Explicit unknown-category flags over frequency bucketing",
     "Eight cities appear only at scoring time. With plain one-hot encoding they become an all-zero "
     "vector, indistinguishable from 'no city at all'.",
     "min_frequency bucketing to create an 'infrequent' catch-all; target encoding; dropping the "
     "categorical columns.",
     "Keep handle_unknown='ignore' and add explicit pickup_is_unknown and delivery_is_unknown "
     "indicators.",
     "Bucketing was measured and rejected: the lowest threshold that creates a bucket at all "
     "collapses 14 of 64 genuine cities, destroying real identity to catch unknowns. The indicator "
     "makes the unknown state learnable at a cost of two columns instead.",
     "The 8 cities affect roughly 6% of scoring rows. Coordinate-derived features remain fully "
     "informative for them regardless of category membership."),

    ("D9. One-hot encoding retained for categoricals",
     "pickup and delivery have 64 levels each, producing 131 sparse columns against 25 dense "
     "features.",
     "Target or ordinal encoding; native categorical handling in CatBoost; hashing.",
     "Plain one-hot encoding, fitted on training rows only.",
     "Target encoding introduces a supervised statistic into preprocessing, which would put fitted "
     "target information inside every CV fold and materially complicate the leakage argument. "
     "One-hot is model-agnostic and provably leakage-free, and the explosion is bounded.",
     "Every model family was compared on the same encoding, so the comparison is like-for-like. "
     "Native categorical handling is listed as future work rather than assumed better."),

    ("D10. Coordinate lookup to serve the reduced December schema",
     "score.py pins the December input to seven columns. Four coordinates, market_index and "
     "quote_signal are absent, yet the pipeline requires all of them.",
     "Skip the deliverable; impute the missing columns with training medians; guess the scenario.",
     "Reconstruct coordinates from a verified city lookup, and market context from a per-date table "
     "built from the scoring set.",
     "The reconstruction is exact rather than approximate: every city maps to exactly one "
     "coordinate pair across both datasets, so the lookup is a fact rather than an estimate. "
     "Median imputation would have flattened the curve and hidden the daily variation the chart "
     "exists to show.",
     "All 31 December dates are present in the scoring set with 163-227 loads each; their daily "
     "market index spans 0.831-1.045, inside the training range of 0.676-1.468."),

    ("D11. Log target",
     "posted_rate is strictly positive and right-skewed - median $2,031, mean $2,374, maximum "
     "$25,533.",
     "Identity target; Box-Cox; square-root transform.",
     "Train on log(posted_rate) and invert with exp.",
     "Log training addresses the skew and makes the model optimise relative rather than absolute "
     "error, which matches how freight is actually priced. It also guarantees strictly positive "
     "predictions after inversion - the constraint score.py enforces - rather than relying on "
     "post-hoc clipping.",
     "In Phase 4 the log-target Ridge beat identity Ridge on MAE ($145.24 against $149.36) and "
     "MAPE (6.27% against 8.65%)."),

    ("D12. Duan smearing correction, tested before adoption",
     "A model fitted to the conditional mean of log(rate) returns the conditional median after "
     "exp(), which sits below the mean for a right-skewed target. The measured bias was +$101.81 "
     "(t = 15.69).",
     "Accept the bias; apply a parametric log-normal correction.",
     "Rescale predictions by the mean of exponentiated training residuals (factor 1.0128).",
     "The parametric correction assumes normally distributed residuals, which the QQ plot rejects "
     "outright. Duan's estimator is non-parametric and makes no distributional assumption. "
     "Critically, it was evaluated before adoption: smearing targets the conditional mean while MAE "
     "is minimised by the median, so the two objectives genuinely conflict and adopting it blind "
     "would have been a coin flip.",
     "On the untouched holdout all three metrics improved: MAE $132.15 to $114.99, RMSE $641.48 to "
     "$636.25, bias $101.81 to $73.37."),

    ("D13. CatBoost selected, with the margin stated honestly",
     "Five tuned model families produced holdout MAE between $132 and $158.",
     "LightGBM ($135.78); HistGradientBoosting ($136.21); RandomForest; XGBoost.",
     "CatBoost, at MAE $132.15 before correction.",
     "It produced the lowest holdout MAE and MAPE. However the top three sit within $4 of each "
     "other on a $2,280 mean prediction, which is inside fold-to-fold variation - so the honest "
     "characterisation is one of three near-equivalent models, not a decisive winner.",
     "CatBoost $132.15, LightGBM $135.78, HistGradientBoosting $136.21. RMSE for all three is "
     "within $2."),

    ("D14. Positive prediction clipping and pipeline persistence",
     "score.py rejects any non-positive predicted_rate, and a model split from its preprocessing "
     "can drift out of sync between training and serving.",
     "Clip only at submission time; persist the model and preprocessor as separate artifacts.",
     "Apply the floor inside the scoring path so every reported metric faces the same constraint as "
     "the submission, and persist preprocessing and model together as one pipeline object.",
     "Evaluating without the floor would report metrics the submission could not achieve. Persisting "
     "one object removes an entire class of serving bug: there is no second artifact to version, "
     "deploy or accidentally mismatch.",
     "The submitted file was verified: 12,000 rows, all finite, all strictly positive, minimum "
     "$201.65, template ordering preserved."),
]

# --------------------------------------------------------------------------- #
# Risks: (risk, likelihood, impact, signal, mitigation)
# --------------------------------------------------------------------------- #
RISKS: list[tuple[str, str, str, str, str]] = [
    ("Covariate drift in market conditions", "High", "High",
     "Daily market_index moves outside the observed 0.676-1.468 band.",
     "Monitor the input distribution weekly; retrain when the band is breached."),
    ("Concept drift in pricing behaviour", "Medium", "High",
     "Realised-versus-predicted error rises while inputs stay in-distribution.",
     "Track rolling MAE against the $114.99 reference; retrain on a rolling window."),
    ("New cities entering the network", "High", "Low",
     "pickup_is_unknown or delivery_is_unknown fires on a rising share of loads.",
     "Already handled: geography carries these rows. Backfill lane history as it accumulates."),
    ("Fuel price shocks", "Medium", "High",
     "Systematic under-prediction across all distance bands simultaneously.",
     "Fuel is not an input feature. A sustained shock requires retraining, not recalibration."),
    ("Holiday and seasonal effects", "High", "Medium",
     "Elevated error concentrated on specific calendar dates.",
     "Cyclical encoding keeps predictions in-support, but no labelled Nov-Dec data exists to learn "
     "holiday effects from."),
    ("Heavy-tail loads", "Certain", "Medium",
     "Already characterised: the worst 1% of loads carry 39.3% of total error.",
     "Route the top percentile by predicted rate to human review."),
    ("Stale market context", "Medium", "High",
     "market_index_is_missing fires, or the daily table is not refreshed.",
     "Alert on missing daily market data; the whole day's predictions degrade together."),
]

# --------------------------------------------------------------------------- #
# Lessons: (lesson, detail)
# --------------------------------------------------------------------------- #
LESSONS: list[tuple[str, str]] = [
    ("The most dangerous bug was silent, not loud",
     "A bare try/except in the inherited preprocessing runner was swallowing a genuine failure and "
     "renaming all 142 feature columns to integers on every run. Nothing crashed and no metric "
     "looked wrong. The lesson generalises: in an ML pipeline, the defects that survive longest are "
     "the ones that still produce a number. Every silent fallback was replaced with an explicit "
     "assertion."),
    ("Recognising the problem structure mattered more than model choice",
     "The single highest-leverage decision was recognising the train and scoring windows do not "
     "overlap, which reframed the task as forward extrapolation. That one observation drove the "
     "split design, the date encoding and the feature set. By contrast, the gap between the top "
     "three model families was $4 - within noise. Framing beat tuning by an order of magnitude."),
    ("Aggregate correlation is not sensitivity",
     "Daily mean market index correlates +0.577 with daily mean rate-per-mile, and an earlier draft "
     "of the analysis described it as the dominant operational signal. The permutation importance "
     "contradicted that, ranking it 15th out-of-sample. The measured elasticity is only 0.139. "
     "Correlation between averages says nothing about the response magnitude, and the claim was "
     "corrected rather than left standing."),
    ("Test a correction before adopting it",
     "Duan smearing was expected to reduce bias but risked worsening MAE, because it targets the "
     "conditional mean while MAE is minimised by the median. Measuring it on the untouched holdout "
     "before adoption converted a coin flip into an evidence-backed 13% improvement - and would "
     "equally have justified rejecting it."),
    ("RMSE could not discriminate between models",
     "Across six model families RMSE stayed pinned between $636 and $654 while MAE ranged from $115 "
     "to $1,149. Selecting on RMSE would have made every model look equivalent. Choosing the metric "
     "that matches the business cost function was a prerequisite for the comparison being "
     "meaningful at all."),
    ("An unexplained result is not a finished result",
     "The near-flat December curve looked like a bug. Rather than shipping it or forcing variation, "
     "the elasticity was measured directly from the data: it implies a +3.2% response where the "
     "model produces +2.7%. The curve is correct, and knowing why is what makes it defensible."),
]
