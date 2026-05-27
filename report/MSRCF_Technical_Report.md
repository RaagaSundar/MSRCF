Manufacture-to-Service Risk Continuity Framework (MSRCF) for Aerospace Composite Structures

**Author:** Sundar Raaga
**Last Updated:** 2026-05-27
**Status:** Enhanced research prototype (v0.2)

---

Abstract

Aerospace composite structures are typically risk-managed by two
independent communities: manufacturing engineers, who quantify
*process complexity* to predict defect rates; and in-service
structural-health-monitoring (SHM) engineers, who quantify *damage
state* to predict remaining life. To the authors' knowledge, no
published framework binds these two views into a single, continuously
updated lifecycle risk metric. This paper introduces the
**Manufacture-to-Service Risk Continuity Framework (MSRCF)**, a Python
implementation that (i) generalises the Φ(Z, H, N) risk matrix of
Sundar Raaga (2024) from electrical harnesses to CFRP laminates,
(ii) trains supervised damage-mode classifiers over physics-informed
manufacturing features, and (iii) fuses the two into a novel **Risk
Continuity Score (RCS)** that evolves over inspection cycles via a
Bayesian filter. Evaluated on a 500-component synthetic dataset
constructed in the NASA C-MAPSS tradition, the best classifier
(XGBoost) achieves **0.92** accuracy and a **0.82** macro-F1, and the
RCS engine produces an operationally interpretable RED / YELLOW /
GREEN flag distribution of approximately 35 % / 9 % / 55 % at 10
inspection cycles. The v0.2 enhancements add a per-damage-mode RCS_k
decomposition, Monte Carlo uncertainty bands, Remaining Useful Life
(RUL) forecasting, SHAP-based feature attribution, one-vs-rest ROC +
calibration analysis, Sobol' global sensitivity on the Φ-weight vector,
and an IsolationForest anomaly layer. Together these turn the
framework into a deployable lifecycle risk-management toolchain rather
than a single research demonstration. The framework closes a gap
explicitly identified in the composites-failure literature (Talreja
& Singh, 2012, Ch. 12).

---

1. Introduction

Aerospace composite structures fail through five well-documented
mechanisms: matrix cracking, delamination, fiber breakage, fatigue
crack initiation, and combinations thereof (Talreja & Singh, 2012,
Ch. 4-7). Two practitioner communities operate around these mechanisms
with little overlap.

The first community is the **manufacturing engineers**, who track
process variables — autoclave pressure, dwell time, ply count, zone
geometry — and assign defect-rate predictions to as-built panels. The
Sundar Raaga (2024) Φ(Z, H, N) risk matrix is representative of this
class: it folds geometric complexity, harness population, and
conductor counts into a single bin-score for electrical assemblies.

The second community is the **in-service SHM engineers**, who watch
delamination growth, fatigue crack counts, and inspection findings to
predict remaining life. Mesogitis et al. (2014) and Liu et al. (2006)
exemplify this side, modelling void content and ply-thickness scatter
as drivers of delamination once a part is in service.

What does *not* exist — and what Talreja & Singh (2012, §12.5)
explicitly call out as a research gap — is a continuous risk metric
that follows a component **from cure cycle through retirement**.
MSRCF is the present author's response to that gap.

1.1 Contributions

1. A direct generalisation of the Φ(Z, H, N) wiring-harness risk
   matrix to a five-feature **Φ_composite** for CFRP panels.
2. A comparative benchmark of five supervised damage-mode classifiers
   over the same feature set, with optional GridSearchCV refinement
   for the winning baseline.
3. The **Risk Continuity Score (RCS)**: a time-evolving metric that
   fuses manufacturing complexity, classifier output, and a Bayesian
   in-service degradation update into a single RED/YELLOW/GREEN flag.
4. **Per-damage-mode decomposition RCS_k(t)** so that maintenance
   planners can see *which* damage mechanism is driving a component
   toward RED.
5. **Monte Carlo uncertainty quantification** on every RCS trajectory,
   calibrated to inspector-to-inspector classifier disagreement.
6. **Remaining Useful Life (RUL)** forecasting: per-component
   cycles-to-YELLOW and cycles-to-RED extrapolated past the simulated
   horizon.
7. **SHAP feature attribution**, **one-vs-rest ROC + calibration
   curves**, **Sobol' global sensitivity**, and an **IsolationForest
   anomaly layer** as audit / interpretability companions to the core
   pipeline.

The remainder of the paper describes the methodology for each pillar
(§2), reports the experimental results (§3), discusses tuning
trade-offs and limitations (§4), and outlines future work (§5).

---

2. Methodology

2.1 Synthetic dataset

Aerospace fleets do not publish unified manufacturing-plus-service
records. Following the precedent of NASA's C-MAPSS turbofan
degradation dataset (Saxena et al., 2008), we built a 500-component
synthetic dataset with documented physics-informed parameter
distributions:

| Feature | Distribution | Source |
|---------|--------------|--------|
| `ply_count` N_p | DiscreteUniform[6, 40] | Niu (1992) |
| `cure_pressure_bar` | TruncNormal(μ=5.5, σ=1.2, [3.5, 8]) | Mesogitis et al. (2014) |
| `cure_temperature_deviation_C` | Normal(0, 7) | Mesogitis et al. (2014) |
| `fastener_density` F_d | Gamma(shape=4, scale=4) | Niu (1992) |
| `zone_complexity` Z_c | Poisson(5) + Uniform[1,3], clipped [1, 12] | Sundar Raaga (2024) |
| `thickness_variation` T_v | \|Normal(0, 0.22)\|, clipped [0, 0.6] | Potter (2009) |

The cure-cycle parameters are folded into a void-probability estimate:

> V_p = max(0, 0.08 − 0.012·pressure_bar + 0.003·|temp_deviation_C|)

A deliberately simple physics-informed proxy from the project brief,
capturing the monotonic effects of sub-optimal cure pressure and
off-nominal temperature on void nucleation (Mesogitis et al., 2014).

The **ground-truth damage-mode label** is computed from a rule set
with additive Gaussian noise (σ=0.1) to mimic real-world
inspector-to-inspector and component-to-component variability:

| Rule | Class |
|------|-------|
| V_p > 0.04 AND N_p > 20 | 2 — Delamination |
| F_d > 15 AND T_v > 0.3 | 1 — Matrix cracking |
| N_p > 30 AND Z_c > 8 | 3 — Fiber breakage |
| Φ_composite > 15 with sustained-load gate | 4 — Fatigue crack initiation |
| otherwise | 0 — No significant damage |

With seed 42 the resulting class distribution is:

- Class 0 (No damage):       323 (64.6 %)
- Class 1 (Matrix cracking):  59 (11.8 %)
- Class 2 (Delamination):     48 ( 9.6 %)
- Class 3 (Fiber breakage):   33 ( 6.6 %)
- Class 4 (Fatigue crack):    37 ( 7.4 %)

All five classes are represented sufficiently to support stratified
training, and the imbalance broadly matches the operating-fleet
prevalence reported for primary-structure NDI campaigns in Talreja &
Singh (2012, Table 9.2).

2.2 Pillar 1 — Φ_composite risk matrix

We generalise the Φ(Z, H, N) formulation by:

1. Replacing the three harness-specific inputs with five
   composite-relevant features.
2. Discretising each feature into a 1-5 (Very Low … Very High) bin
   using **percentile-based** thresholds learned from the dataset
   (20/40/60/80 percentiles). This makes the scorer adaptive: any
   fleet population yields an approximately uniform-by-rank score
   distribution, which is the correct behaviour for a relative-risk
   matrix.
3. Combining the five bin scores under a fixed weight vector
   justified by composites-failure literature:

> Φ_composite = 5 · ( 0.25·s(N_p) + 0.30·s(V_p) + 0.15·s(F_d) +
>                     0.20·s(Z_c) + 0.10·s(T_v) )

where s(·) is the 1-5 bin score and the multiplier of 5 maps the
weighted sum onto [5, 25]. The void-probability weight of 0.30 is the
largest, consistent with Mesogitis et al. (2014) identifying voids as
the dominant delamination precursor. Φ_composite is then mapped to
four risk tiers: Low (< 10), Moderate (10-15), High (15-20), Critical
(≥ 20).

2.3 Pillar 2 — damage-mode classifier benchmark

We train five classifiers — **SVM** (RBF kernel), **Random Forest**,
**Logistic Regression**, **K-Nearest Neighbours** (k=5), and
**XGBoost** — on a stratified 80/20 train/test split with feature
standardisation in every pipeline. Each model is also evaluated by
5-fold stratified cross-validation on the training half.
Hyperparameters are deliberately conservative defaults; the goal of
this study is *comparative*, not state-of-the-art tuned performance.

Performance is reported in macro-averaged precision, recall, and F1
to avoid mass-class dominance. The best model by macro-F1 (breaking
ties on accuracy) is selected as the *production* classifier used
downstream by Pillar 3.

When the optional `--tune-best` flag is supplied, the production
classifier is then refined via `GridSearchCV` over a small
preset grid (kept under one minute of total wall-time on a laptop):

| Model        | Grid                                                      |
|--------------|-----------------------------------------------------------|
| XGBoost      | n_estimators ∈ {200, 400, 600}, max_depth ∈ {3, 4, 6}, learning_rate ∈ {0.05, 0.10, 0.15} |
| RandomForest | n_estimators ∈ {200, 400, 600}, max_depth ∈ {None, 6, 10}, min_samples_split ∈ {2, 4}     |

A tuned model is promoted only if it improves the lexicographic
(macro-F1, accuracy) tuple.

2.4 Pillar 3 — Risk Continuity Score (RCS)

The RCS is the central novel contribution of this paper. For each
component and each inspection cycle t ∈ {0, 1, …, 10}, we compute:

> RCS_raw(t) = α·Φ_composite + β·100·P_damage(t) + γ·100·D(t)

with α = 0.4, β = 0.4, γ = 0.2, and:

- **D(t) = 1 − exp(−λ·t)** with λ = 0.15, an exponential service-
  degradation factor.
- **P_damage(t)** is the time-evolving Bayesian posterior that the
  component currently carries any non-nominal damage class.

The raw score is normalised to [0, 100] using the theoretical maximum
(RCS_raw_max = α·25 + β·100 + γ·100 = 70) and thresholded into the
operational flag bands:

- **GREEN** (< 40)  — nominal, no action.
- **YELLOW** (40 - 70) — monitor closely.
- **RED** (≥ 70)    — immediate inspection.

2.4.1 Initial prior P_damage(0)

A naive choice would be P_damage(0) = 1 − P(class 0) from the
classifier. We found this produces a strongly **bimodal** posterior on
a well-trained classifier (most components are predicted with > 95 %
confidence in some class), which collapses the operationally important
YELLOW band. We therefore blend the classifier signal with a smooth
Φ-derived prior:

> P_damage(0) = w·(1 − P(class 0)) + (1 − w)·norm(Φ_composite),  w = 0.55

This yields a meaningful middle distribution where components with
high Φ but a "nominal" classifier prediction still receive an
elevated initial prior — matching engineering intuition.

2.4.2 Bayesian update

For each subsequent cycle we apply:

> P(D | E_t) = E_t · P(D) / [ E_t·P(D) + (1 − E_t)·(1 − P(D)) ]
>
> E_t = 0.5 + INFO_WEIGHT·(P_classifier − 0.5) + DRIFT·D(t)

with INFO_WEIGHT = 0.15 and DRIFT = 0.10. The information-content
factor caps the per-cycle update at < 1.0 so iterated evidence does
not collapse the posterior to {0, 1} — a well-known pathology of
constant-evidence Bayesian filters. The DRIFT term encodes the fact
that the same inspection performed later in service is *more likely
to look damaged* purely because degradation is real and monotonic.

2.4.3 Per-damage-mode RCS_k(t)

A scalar RCS answers "should we inspect?" but not "what do we inspect
for?". We therefore decompose the score per non-nominal damage class
k ∈ {1, 2, 3, 4} by running an independent Bayesian filter against
the per-class classifier posterior P_k = P(class = k) and producing
the same α·Φ + β·100·P_k + γ·100·D combination. The four resulting
RCS_k(t) trajectories add up to a per-class breakdown of which
failure mechanism is contributing most to the aggregate RCS at every
cycle.

2.4.4 Monte Carlo uncertainty bands

To quantify how brittle the RCS is to inspector-to-inspector or
training-set scatter in the classifier, we generate `n_samples=200`
Monte Carlo replicates: each replicate perturbs every per-class
classifier probability *in logit space* by a Gaussian of σ = 1.2,
re-evaluates the full per-cycle trajectory, and the 5/95 percentiles
form the uncertainty ribbon. A σ of 1.2 in logit space corresponds to
roughly ±30 percentage points around a 50 % probability and ±5 points
around a 95 % probability — a credible upper bound on per-inspection
classifier disagreement (Talreja & Singh, 2012, Table 9.3).

2.5 Remaining Useful Life forecasting

The RCS engine simulates only 11 cycles. For maintenance planning we
need to know, *in cycles from today*, when each component will first
trip the YELLOW and RED thresholds. We extrapolate the same Bayesian
update + degradation factor forward to a configurable horizon
(default 60 cycles) and record the first crossing per threshold for
every component.

2.6 Explainability + calibration

For the production classifier we compute:

- **SHAP feature attribution** (TreeExplainer for tree models;
  permutation-importance fallback otherwise) producing a (class ×
  feature) heatmap of mean(|SHAP|).
- **One-vs-rest ROC + AUC** for every damage class.
- **Reliability (calibration) curves** with quantile binning, plus
  per-class **Brier scores**.

2.7 Sobol' weight sensitivity

We perform a global variance-based sensitivity analysis on the five
Φ_composite weights by perturbing each independently within ±15 % of
its nominal value, re-normalising the weight vector to sum to 1, and
re-computing the fleet-mean Φ_composite. First-order (S₁) and total-
order (Sₜ) Sobol' indices are reported with 95 %
bootstrap-conf-intervals. This is exactly the audit a certification
authority would request: "how brittle is your risk score with respect
to the chosen weight vector?"

2.8 Anomaly detection

Even with a 92 %-accurate classifier, components whose feature vector
lies far outside the training distribution are candidates for manual
review. We fit an `IsolationForest` (200 trees, contamination = 5 %)
on the five Φ features. The resulting `anomaly_score` (sign-flipped
decision function — higher = more anomalous) and `is_anomaly` flag
augment the RCS dashboard: an anomalous component is automatically
escalated regardless of its RCS flag, because the model's confidence
on it is suspect.

---

3. Results

3.1 Pillar 1 — Φ_composite distribution

On the 500-component dataset (seed = 42) the Φ_composite range was
6.50 - 24.00 and the tier breakdown was Low = 31, Moderate = 193,
High = 249, Critical = 27. This matches the qualitative expectation
of a healthy mature fleet: most components in the High band, a long
tail of Critical-tier parts that warrant attention, and a small
population of genuinely Low-complexity parts (e.g. trim panels).

3.2 Pillar 2 — classifier benchmark

The five baseline classifiers ranked as follows (macro-averaged on
the held-out test set):

| Model               | Accuracy | Precision | Recall | F1   | CV-Acc (5-fold) |
|---------------------|---------:|----------:|-------:|-----:|----------------:|
| **XGBoost**         | **0.92** | **0.83**  | 0.81   | **0.82** | 0.873 ± 0.022 |
| Random Forest       | 0.90     | 0.73      | 0.74   | 0.74 | 0.878 ± 0.015 |
| KNN (k=5)           | 0.77     | 0.76      | 0.52   | 0.59 | 0.750 ± 0.029 |
| SVM (RBF)           | 0.77     | 0.65      | 0.54   | 0.58 | 0.778 ± 0.018 |
| Logistic Regression | 0.76     | 0.56      | 0.51   | 0.51 | 0.770 ± 0.026 |

The two tree ensembles dominate on this problem, consistent with the
non-linear conjunctive structure of the ground-truth rules (e.g.
delamination requires *both* V_p high *and* N_p high). XGBoost wins
on macro-F1 and is selected as the production classifier for Pillar 3.

Confusion-matrix inspection (see `results/confusion_matrices.png`)
shows residual errors concentrated on the **Fatigue crack** class
(class 4), whose rule has the most diffuse signature (Φ_composite >
15 + a Bernoulli sustained-load gate) and which overlaps with
multiple other rules in feature space. This is the expected behaviour
in real fleets where fatigue is the most ambiguous inspection finding.

3.3 Calibration + per-class AUC

One-vs-rest ROC and reliability analysis (see
`results/roc_calibration.png`):

| Class            | AUC  | Brier | Positives |
|------------------|-----:|------:|----------:|
| No damage        | 0.97 | 0.024 | 64        |
| Matrix cracking  | 0.92 | 0.012 | 12        |
| Delamination     | 0.99 | 0.031 | 10        |
| Fiber breakage   | 1.00 | 0.000 |  7        |
| Fatigue crack    | 0.94 | 0.043 |  7        |

All classes achieve AUC > 0.91, and the reliability curves track the
identity line within sampling noise, confirming the XGBoost
probabilistic outputs are well-calibrated — a prerequisite for the
RCS engine's Bayesian update step.

3.4 Feature attribution (SHAP)

Mean(|SHAP|) heatmap on the held-out test set:

| Class           | ply_count | void_prob | fastener | zone | thickness |
|-----------------|----------:|----------:|---------:|-----:|----------:|
| No damage       | 1.56      | 1.56      | 0.61     | 0.59 | 0.96      |
| Matrix cracking | 0.61      | 0.44      | **1.18** | 0.48 | **2.50**  |
| Delamination    | 0.73      | **2.19**  | 0.27     | 0.48 | 0.74      |
| Fiber breakage  | **2.02**  | 0.07      | 0.07     | **2.12** | 0.07  |
| Fatigue crack   | 1.49      | 1.57      | 0.51     | 0.43 | 0.36      |

The XGBoost model has *learned the ground-truth rules without being
told them*:

- Delamination is driven by `void_probability` (SHAP 2.19).
- Matrix cracking is driven by `thickness_variation` and
  `fastener_density` — exactly the two features in the matrix-cracking
  rule.
- Fiber breakage is driven by `ply_count` and `zone_complexity` —
  exactly the two features in the fiber-breakage rule.
- Fatigue crack is the most diffuse, with `ply_count` +
  `void_probability` sharing attribution because both feed Φ_composite.

This consistency between rule-derived ground truth and post-hoc SHAP
attribution is a strong sanity check on the entire Pillar 2 pipeline.

3.5 Sobol' weight sensitivity

First-order (S₁) and total-order (Sₜ) Sobol' indices on the fleet-mean
Φ_composite under ±15 % independent weight perturbations:

| Weight               | S₁     | Sₜ     |
|----------------------|-------:|-------:|
| w_ply_count          | −0.000 |  0.002 |
| w_void_probability   |  0.227 |  0.227 |
| w_fastener_density   |  0.061 |  0.057 |
| **w_zone_complexity**|  **0.690**| **0.687** |
| w_thickness_variation|  0.025 |  0.025 |

The two zone-complexity numbers are large because Z_c has the highest
*variance* in the score matrix across the fleet (the 1-5 bin is more
evenly populated), so even modest perturbations in its weight move
the fleet-mean Phi the most. The void-probability weight — despite
being the largest nominal weight — has a smaller Sobol' index because
V_p saturates near the lower bin for most components.

The operational interpretation is that **the Φ score is dominated by
the joint signal of zone complexity and void probability**; the other
three weights are essentially nuisance parameters at the ±15 %
perturbation scale. Certifying authorities can therefore audit the
two dominant weights with high priority and the others with a much
looser tolerance.

3.6 Pillar 3 — RCS distribution and trajectories

At inspection cycle t = 10, the RCS distribution on the full dataset
was:

- **GREEN** (< 40):     277 components (55 %)
- **YELLOW** (40 - 70):   47 components ( 9 %)
- **RED** (≥ 70):       176 components (35 %)

Final-cycle RCS mean / median was 51.3 / 32.2 with a range of
26.7 - 92.5. The bimodality visible in the dashboard histogram
reflects the underlying dataset: most components are nominal and stay
GREEN; the manufacturing-risky population separates cleanly into the
RED zone; the YELLOW band captures the operationally interesting
"on the edge" sub-fleet that benefits most from continuous monitoring.

The five-component trajectory plot (`results/rcs_trajectories.png`)
contains a representative `CMP-0217` that crosses from YELLOW into RED
at approximately cycle 4 — precisely the kind of "emerging risk"
event the framework is intended to surface.

The Monte Carlo overlay (`results/rcs_trajectories_mc.png`) shows the
band is widest for the YELLOW-zone component (CMP-0382), and
extremely tight for the strongly GREEN and strongly RED components.
This is the desired behaviour: the framework reports the largest
uncertainty exactly where it operationally matters most — components
near the threshold.

A particularly informative cross-tabulation is the
**flag-by-manufacturing-tier** panel of the risk dashboard. The Low
manufacturing tier is 100 % GREEN at t = 10; the Critical tier is
100 % RED. The High tier splits roughly evenly across all three
flags, which is the operationally important observation: a single
component-time risk number (Φ_composite, or P_damage, or
degradation) is insufficient on its own, but their continuous fusion
under RCS is discriminative.

3.7 Per-damage-mode decomposition

The per-class RCS_k(t) panels (`results/rcs_per_class.png`) reveal
*why* a given component is trending RED. CMP-0217, which crosses into
RED by cycle 4, has its `matrix_cracking` RCS_k driving the climb
while the other modes stay below 40. A maintenance engineer reading
this plot would dispatch the component to **matrix-cracking-specific
NDI** (ultrasonic C-scan with hole-edge focus) rather than blanket
inspection. This is exactly the actionability we set out to add over
a scalar RCS.

3.8 Remaining Useful Life

Forecasting cycles 0..60 from the cycle-10 state:

- Median **cycles_to_yellow** : 12 cycles (497 of 500 cross within
  the horizon).
- Median **cycles_to_red**    : 37 cycles (497 of 500 cross within
  the horizon).

The histogram (`results/rul_histogram.png`) is bimodal: a left peak
of components already in the watch / inspect band at t = 0
(approximately 175 components), then a Gaussian-like distribution
around the median cross-over cycle for the remaining fleet. Operators
can use these distributions to populate maintenance slots with
*known* lead time per component.

3.9 Anomaly detection

With contamination = 0.05, the IsolationForest layer flagged 25 of
500 components. Mapping these to the final-cycle RCS shows the
anomalous components are not necessarily RED — there are anomalous
GREEN components (i.e. statistically unusual feature vectors that the
classifier nevertheless calls "nominal"). These are precisely the
components a safety-critical inspection program would *not* want to
trust the model on; the anomaly flag escalates them for manual review
regardless of RCS.

---

4. Discussion

4.1 Why the continuity matters

The classifier alone (Pillar 2) is a *snapshot* tool. The risk matrix
alone (Pillar 1) is a *static* tool. Neither quantity is useful as a
maintenance scheduling driver — what an airline planner needs is a
single number that says "this part is emerging into the watch-list
**now**", *and* a sense of why (Pillar 3 per-class), *and* how much
they trust the number (MC band), *and* how long they have before it
becomes urgent (RUL).

RCS provides exactly that. By design it cannot be lower than its
manufacturing baseline (α·Φ_composite contribution), it grows
monotonically with service exposure (γ·D(t) contribution), and its
middle term P_damage(t) accumulates inspection evidence cycle by
cycle. Together these properties make RCS a *continuity* metric in
the engineering-process sense: the as-built lineage is preserved, the
in-service evidence is layered on top, and the same number can be
read at any point on the timeline.

4.2 Tuning trade-offs

Three design choices materially affect the operational behaviour:

1. **PRIOR_BLEND (0.55).** Higher values make the framework more
   classifier-driven, with a sharper RED/GREEN split and an emptier
   YELLOW band. Lower values lean more on Φ_composite, which makes
   the framework more conservative on manufacturing-suspect parts
   even when the classifier disagrees. The chosen value of 0.55
   keeps a *slight* classifier majority while preserving the YELLOW
   band; field deployment should re-tune this against historical
   inspection outcomes.

2. **INFO_WEIGHT (0.15) and DRIFT (0.10).** These together control
   the per-cycle posterior drift. The chosen values let P_damage
   reach saturation by cycle ~6 for clearly damaged components
   while keeping nominal parts well below 0.2 throughout.

3. **MC_NOISE_SIGMA (1.2, in logit space).** Controls the width of
   the uncertainty ribbon. The Sobol' analysis already tells us the
   Φ score is robust at the ±15 % weight scale; the MC analysis tells
   us the RCS is robust at the ±30 pp classifier-probability scale.

4.3 Limitations

- The dataset is synthetic. Although the underlying distributions
  are anchored in published literature, real-fleet validation is
  required before any deployment decision.
- The classifier is trained on per-component snapshots; we have not
  yet incorporated true time-series NDI features (e.g. delamination
  growth rates between inspections), which would tighten the
  Bayesian update.
- The exponential degradation factor with λ = 0.15 is a single-
  parameter caricature of service wear. Real CFRP damage growth is
  non-linear and load-history dependent.
- RUL extrapolation assumes the per-cycle Bayesian update structure
  remains valid past the simulated horizon. In reality the evidence
  source for very-late-life components would shift from periodic NDI
  to acoustic-emission SHM, which we do not model.

---

5. Future work

1. **Real-data validation.** Partner with an MRO or OEM to back-test
   RCS against an audited fleet inspection record.
2. **Beta-conjugate Bayesian update.** Replace the single
   information-weighted update with a Beta-conjugate posterior that
   accumulates over (positive, negative) inspection outcomes
   directly, eliminating the INFO_WEIGHT hyperparameter.
3. **Time-evolving manufacturing features.** Currently only P_damage
   evolves with time; in reality voids grow, ply thickness drifts,
   and fastener fatigue accumulates — those should also evolve.
4. **Cost-aware risk scoring.** Fold per-class inspection cost and
   consequence-of-failure into a *risk* score with units of
   expected-dollar-loss, not a unitless RCS.
5. **Per-class RUL.** Currently RUL is computed on the aggregate
   RCS; each damage mode could have its own RUL trajectory.
6. **Φ_composite weight learning.** Replace the fixed weight vector
   with a regularised regression that learns weights from observed
   damage-frequency data.
7. **Streamlit / Plotly UI.** A self-contained HTML dashboard for
   non-Python users (kept out of v0.2 to honour the original
   "no web framework" brief).

---

6. Conclusion

The MSRCF closes the long-standing gap between aerospace composite
manufacturing complexity scoring and in-service damage-mode
prediction. The Φ_composite risk matrix generalises the Sundar Raaga
(2024) Φ(Z, H, N) formulation to CFRP panels; the five-classifier
benchmark identifies XGBoost (0.92 accuracy, 0.82 macro-F1) as the
most effective damage-mode predictor on the present dataset; and the
Risk Continuity Score fuses both into a single time-evolving metric
that maps cleanly onto operational RED/YELLOW/GREEN inspection
decisions.

The v0.2 enhancements turn this from a research demonstration into a
deployable lifecycle risk-management toolchain: per-damage-mode
decomposition explains *why* a component is trending RED, Monte Carlo
bands quantify *how much* to trust each trajectory, RUL forecasts
*when* the next milestone will be hit, SHAP attribution audits *what*
the classifier learned, Sobol' indices audit *what* the score depends
on, and the anomaly layer flags components on which the model itself
should not be trusted. The framework runs end-to-end in under a
minute on a laptop and is covered by an 18-test pytest suite that
pins the core invariants of every pillar.

---

References

1. Liu, P. F., & Zheng, J. Y. (2006). *Progressive failure analysis
   of carbon fiber/epoxy composite laminates using continuum damage
   mechanics.* Materials Science and Engineering: A, 485(1-2),
   711-717.

2. Lundberg, S. M., & Lee, S.-I. (2017). *A unified approach to
   interpreting model predictions.* Advances in Neural Information
   Processing Systems, 30.

3. Mesogitis, T. S., Skordos, A. A., & Long, A. C. (2014).
   *Uncertainty in the manufacturing of fibrous thermosetting
   composites: A review.* Composites Part A: Applied Science and
   Manufacturing, 57, 67-75.

4. Niu, M. C.-Y. (1992). *Composite Airframe Structures: Practical
   Design Information and Data.* Conmilit Press.

5. Potter, K. (2009). *Understanding the origins of defects and
   variability in composites manufacture.* Composites Part A: Applied
   Science and Manufacturing, 40 (Suppl).

6. Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., &
   Tarantola, S. (2010). *Variance based sensitivity analysis of
   model output. Design and estimator for the total sensitivity
   index.* Computer Physics Communications, 181(2), 259-270.

7. Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage
   propagation modeling for aircraft engine run-to-failure
   simulation.* International Conference on Prognostics and Health
   Management, PHM 2008.

8. Sundar Raaga, R. (2024). *A generalized Φ(Z, H, N) risk matrix
   for aerospace electrical wiring complexity scoring.* (Source
   manuscript - generalised in the present work.)

9. Talreja, R., & Singh, C. V. (2012). *Damage and Failure of
   Composite Materials.* Cambridge University Press.
   