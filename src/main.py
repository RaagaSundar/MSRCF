"""
main.py
=======

End-to-end orchestrator for the Manufacture-to-Service Risk Continuity
Framework (MSRCF) pipeline (enhanced v0.2).

Pipeline stages:

    1. Generate synthetic CFRP dataset                  (data_generator)
    2. Fit Phi_composite risk-matrix scorer             (risk_matrix)
    3. Train + evaluate 5 damage classifiers            (damage_predictor)
    3b. (optional) Tune the best baseline               (--tune-best)
    4. Compute RCS trajectories + per-class + MC band   (rcs_engine)
    5. Forecast Remaining Useful Life                    (rul_predictor)
    6. Anomaly detection                                 (anomaly_detector)
    7. Explainability + calibration                      (explainability)
    8. Sobol sensitivity on Phi weights                  (sensitivity)
    9. Render figures + summary                          (dashboard)

CLI:

    python src/main.py [--n-components N] [--seed S] [--tune-best]
                       [--mc-samples M] [--skip-shap] [--skip-sobol]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import anomaly_detector             # noqa: E402
import config as cfg                # noqa: E402
import damage_predictor as dp       # noqa: E402
import dashboard                    # noqa: E402
import data_generator               # noqa: E402
import explainability               # noqa: E402
import rcs_engine                   # noqa: E402
import risk_matrix                  # noqa: E402
import rul_predictor                # noqa: E402
import sensitivity                  # noqa: E402


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def _section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MSRCF end-to-end pipeline.",
    )
    parser.add_argument(
        "--n-components", type=int, default=500,
        help="Number of synthetic components to generate (default 500).",
    )
    parser.add_argument(
        "--seed", type=int, default=cfg.RANDOM_SEED,
        help=f"RNG seed (default {cfg.RANDOM_SEED}).",
    )
    parser.add_argument(
        "--tune-best", action="store_true",
        help="Run a GridSearchCV refinement on the best baseline model.",
    )
    parser.add_argument(
        "--mc-samples", type=int, default=cfg.MC_SAMPLES,
        help=f"Monte Carlo samples for the RCS uncertainty band "
             f"(default {cfg.MC_SAMPLES}).",
    )
    parser.add_argument(
        "--skip-shap", action="store_true",
        help="Skip the SHAP explainability stage.",
    )
    parser.add_argument(
        "--skip-sobol", action="store_true",
        help="Skip the Sobol sensitivity analysis stage.",
    )
    return parser.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> dict:
    """End-to-end execution of the enhanced MSRCF pipeline."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # 1) Synthetic dataset
    # -----------------------------------------------------------------
    _section("[1/9] Generating synthetic CFRP dataset")
    dataset_path = os.path.join(DATA_DIR, "msrcf_synthetic_dataset.csv")
    df = data_generator.generate_dataset(
        n_components=args.n_components, output_csv=dataset_path, seed=args.seed
    )
    print(f"  components generated : {len(df)}")
    print(f"  CSV written          : {dataset_path}")
    print("  damage_mode counts   :")
    counts = df["damage_mode"].value_counts().sort_index()
    for cls, ct in counts.items():
        print(
            f"    class {cls} "
            f"({dp.DAMAGE_CLASS_NAMES[int(cls)]:<18}) : {ct}"
        )

    # -----------------------------------------------------------------
    # 2) Phi_composite risk-matrix scoring (Pillar 1)
    # -----------------------------------------------------------------
    _section("[2/9] Pillar 1 - Phi_composite risk matrix")
    rm_model, scored_df = risk_matrix.fit_and_score(df)
    print("  weights:")
    for k, v in risk_matrix.get_feature_weights().items():
        print(f"    {k:<22} -> {v:.2f}")
    print(
        f"  Phi_composite range : "
        f"{scored_df['phi_composite'].min():.2f} - "
        f"{scored_df['phi_composite'].max():.2f}"
    )
    for tier, ct in (
        scored_df["risk_tier"]
        .value_counts()
        .reindex(["Low", "Moderate", "High", "Critical"])
    ).items():
        print(f"    {str(tier):<10} : {int(ct)}")
    scored_csv = os.path.join(DATA_DIR, "msrcf_scored_dataset.csv")
    scored_df.to_csv(scored_csv, index=False)

    # -----------------------------------------------------------------
    # 3) Damage-mode classifiers (Pillar 2)
    # -----------------------------------------------------------------
    _section("[3/9] Pillar 2 - damage-mode classifiers")
    results, splits = dp.train_and_evaluate(df)
    summary = dp.results_to_dataframe(results)
    print(summary.to_string(index=False))
    best = dp.select_best_model(results)
    print(
        f"\n  >> best baseline: {best.name}  "
        f"(accuracy={best.accuracy:.3f}, macro-F1={best.f1_macro:.3f})"
    )

    if args.tune_best:
        _section("[3b/9] Hyperparameter tuning on best baseline")
        grid_map = {
            "XGBoost": cfg.XGB_TUNING_GRID,
            "RandomForest": cfg.RF_TUNING_GRID,
        }
        grid = grid_map.get(best.name)
        if grid is None:
            print(
                f"  (no tuning grid configured for {best.name} - skipping)"
            )
        else:
            tuned = dp.tune_model(
                name=best.name,
                base_estimator=best.estimator,
                X_train=splits["X_train"],
                y_train=splits["y_train"],
                X_test=splits["X_test"],
                y_test=splits["y_test"],
                param_grid=grid,
            )
            print(
                f"  tuned    : accuracy={tuned.accuracy:.3f}, "
                f"macro-F1={tuned.f1_macro:.3f}"
            )
            if (tuned.f1_macro, tuned.accuracy) > (best.f1_macro, best.accuracy):
                print(
                    f"  >> tuned model improves the baseline; "
                    f"promoting {tuned.name} to production"
                )
                results[tuned.name] = tuned
                best = tuned

    summary_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    dp.results_to_dataframe(results).to_csv(summary_csv, index=False)

    # -----------------------------------------------------------------
    # 4) RCS trajectory + per-class + Monte Carlo uncertainty
    # -----------------------------------------------------------------
    _section("[4/9] Pillar 3 - RCS + per-class decomposition + MC band")
    X_full = df[list(dp.FEATURE_COLUMNS)].to_numpy(dtype=float)
    class_labels = best.class_labels
    init_probs = dp.predict_probabilities(best.estimator, X_full, class_labels)

    trajectory = rcs_engine.compute_rcs_trajectory(
        component_ids=df["component_id"].tolist(),
        phi_composite=scored_df["phi_composite"].to_numpy(),
        initial_class_probabilities=init_probs,
        class_labels=class_labels,
    )
    print(f"  cycles simulated      : {trajectory.cycles[0]}..{trajectory.cycles[-1]}")
    print(f"  per-class shapes      : "
          f"rcs_per_class={trajectory.rcs_per_class.shape}")
    final_rcs = trajectory.rcs_normalised[-1]
    print(f"  final RCS range       : {np.min(final_rcs):.2f} - {np.max(final_rcs):.2f}")

    # Monte Carlo bands
    median, lower, upper = rcs_engine.compute_rcs_mc_band(
        component_ids=df["component_id"].tolist(),
        phi_composite=scored_df["phi_composite"].to_numpy(),
        initial_class_probabilities=init_probs,
        class_labels=class_labels,
        n_samples=args.mc_samples,
        noise_sigma=cfg.MC_NOISE_SIGMA,
        ci_low=cfg.MC_CI_LOW,
        ci_high=cfg.MC_CI_HIGH,
        seed=args.seed,
    )
    trajectory.rcs_lower = lower
    trajectory.rcs_upper = upper
    mean_width = float(np.mean(upper - lower))
    print(
        f"  MC samples            : {args.mc_samples} "
        f"(mean 5-95 band width: {mean_width:.2f} RCS points)"
    )

    rcs_long = trajectory.to_long_dataframe()
    rcs_long_csv = os.path.join(DATA_DIR, "msrcf_rcs_trajectories.csv")
    rcs_long.to_csv(rcs_long_csv, index=False)

    flag_counts = pd.Series(trajectory.flag[-1]).value_counts()
    print("  flag distribution     :")
    for f in ["GREEN", "YELLOW", "RED"]:
        print(f"    {f:<6} : {int(flag_counts.get(f, 0))}")

    # -----------------------------------------------------------------
    # 5) Remaining Useful Life forecast
    # -----------------------------------------------------------------
    _section("[5/9] Remaining Useful Life forecast")
    nominal_idx = class_labels.index(0)
    classifier_damage = 1.0 - init_probs[:, nominal_idx]
    rul = rul_predictor.forecast_rul(
        component_ids=df["component_id"].tolist(),
        phi_composite=scored_df["phi_composite"].to_numpy(),
        last_p_damage=trajectory.p_damage[-1],
        classifier_damage=classifier_damage,
        last_cycle=trajectory.cycles[-1],
        horizon_cycle=cfg.RUL_MAX_CYCLE,
    )
    rul_df = rul.to_dataframe()
    rul_csv = os.path.join(DATA_DIR, "msrcf_rul_forecast.csv")
    rul_df.to_csv(rul_csv, index=False)
    yellow_med = rul_df["cycles_to_yellow"].dropna().median()
    red_med = rul_df["cycles_to_red"].dropna().median()
    n_y = int(rul_df["cycles_to_yellow"].notna().sum())
    n_r = int(rul_df["cycles_to_red"].notna().sum())
    print(
        f"  cycles_to_yellow      : median={yellow_med:.1f} "
        f"(n={n_y}/{len(rul_df)} cross within {cfg.RUL_MAX_CYCLE} cycles)"
    )
    print(
        f"  cycles_to_red         : median={red_med:.1f} "
        f"(n={n_r}/{len(rul_df)} cross within {cfg.RUL_MAX_CYCLE} cycles)"
    )

    # -----------------------------------------------------------------
    # 6) Anomaly detection
    # -----------------------------------------------------------------
    _section("[6/9] Anomaly detection")
    _, _, anomaly_df = anomaly_detector.fit_anomaly_detector(
        scored_df,
        list(dp.FEATURE_COLUMNS),
        contamination=cfg.ISO_CONTAMINATION,
        seed=args.seed,
    )
    anomaly_csv = os.path.join(DATA_DIR, "msrcf_anomaly_flags.csv")
    anomaly_df[["component_id", "phi_composite", "anomaly_score", "is_anomaly"]].to_csv(
        anomaly_csv, index=False
    )
    n_anom = int(anomaly_df["is_anomaly"].sum())
    print(
        f"  anomalies flagged     : {n_anom} / {len(anomaly_df)} "
        f"(contamination={cfg.ISO_CONTAMINATION})"
    )

    # -----------------------------------------------------------------
    # 7) Explainability + calibration
    # -----------------------------------------------------------------
    _section("[7/9] Explainability + calibration")
    if not args.skip_shap:
        shap_path, importance_df = explainability.shap_feature_importance(
            best.estimator,
            X_background=splits["X_train"],
            X_explain=splits["X_test"],
            feature_names=list(dp.FEATURE_COLUMNS),
            class_names=dp.DAMAGE_CLASS_NAMES,
            output_path=os.path.join(RESULTS_DIR, "feature_attribution.png"),
        )
        print(f"  feature attribution   : {shap_path}")
        importance_df.to_csv(
            os.path.join(RESULTS_DIR, "feature_attribution.csv")
        )
    else:
        print("  (--skip-shap; skipping)")

    roc_path, roc_df = explainability.roc_and_calibration(
        best.estimator,
        X_test=splits["X_test"],
        y_test=splits["y_test"],
        class_labels=class_labels,
        class_names=dp.DAMAGE_CLASS_NAMES,
        output_path=os.path.join(RESULTS_DIR, "roc_calibration.png"),
    )
    print(f"  ROC + calibration     : {roc_path}")
    roc_df.to_csv(os.path.join(RESULTS_DIR, "roc_calibration.csv"), index=False)
    print(roc_df.to_string(index=False))

    # -----------------------------------------------------------------
    # 8) Sobol sensitivity on Phi weights
    # -----------------------------------------------------------------
    _section("[8/9] Sobol sensitivity on Phi_composite weights")
    if not args.skip_sobol:
        score_cols = [f"score_{f}" for f in risk_matrix.FEATURE_ORDER]
        score_matrix = scored_df[score_cols].to_numpy(dtype=float)
        sobol_path, sobol_df = sensitivity.sobol_phi_weight_sensitivity(
            score_matrix=score_matrix,
            feature_names=list(risk_matrix.FEATURE_ORDER),
            nominal_weights=np.array(cfg.PHI_WEIGHTS),
            output_path=os.path.join(RESULTS_DIR, "sobol_sensitivity.png"),
            n_base=cfg.SOBOL_N_BASE,
            weight_scale=cfg.SOBOL_WEIGHT_SCALE,
            seed=args.seed,
        )
        print(f"  Sobol indices figure  : {sobol_path}")
        sobol_df.to_csv(os.path.join(RESULTS_DIR, "sobol_indices.csv"), index=False)
        print(sobol_df.to_string(index=False))
    else:
        print("  (--skip-sobol; skipping)")

    # -----------------------------------------------------------------
    # 9) Standard figures
    # -----------------------------------------------------------------
    _section("[9/9] Rendering figures")
    cm_path = dashboard.plot_confusion_matrices(
        results, os.path.join(RESULTS_DIR, "confusion_matrices.png")
    )
    bar_path = dashboard.plot_model_comparison(
        results, os.path.join(RESULTS_DIR, "model_comparison.png")
    )
    rcs_path = dashboard.plot_rcs_trajectories(
        trajectory, os.path.join(RESULTS_DIR, "rcs_trajectories.png")
    )
    mc_path = dashboard.plot_rcs_trajectories_with_uncertainty(
        trajectory, os.path.join(RESULTS_DIR, "rcs_trajectories_mc.png")
    )
    pc_path = dashboard.plot_rcs_per_class(
        trajectory, dp.DAMAGE_CLASS_NAMES,
        os.path.join(RESULTS_DIR, "rcs_per_class.png"),
    )
    rul_path = dashboard.plot_rul_histogram(
        rul_df, os.path.join(RESULTS_DIR, "rul_histogram.png")
    )
    dash_path = dashboard.plot_risk_dashboard(
        scored_df, trajectory,
        os.path.join(RESULTS_DIR, "risk_dashboard.png"),
    )
    final_rcs_series = pd.Series(
        trajectory.rcs_normalised[-1], index=trajectory.component_ids
    )
    anom_path = dashboard.plot_anomaly_scatter(
        anomaly_df,
        output_path=os.path.join(RESULTS_DIR, "anomaly_scatter.png"),
        final_rcs=final_rcs_series,
    )
    for p in (cm_path, bar_path, rcs_path, mc_path, pc_path, rul_path,
              dash_path, anom_path):
        print(f"  saved : {p}")

    _section("Summary")
    print(
        f"  dataset rows                  : {len(df)}\n"
        f"  Pillar 1 Phi range            : "
        f"{scored_df['phi_composite'].min():.2f} - "
        f"{scored_df['phi_composite'].max():.2f}\n"
        f"  Pillar 2 best classifier      : {best.name} "
        f"(macro-F1={best.f1_macro:.3f}, acc={best.accuracy:.3f})\n"
        f"  Pillar 3 RED-flag components  : "
        f"{int(flag_counts.get('RED', 0))} / {len(df)}\n"
        f"  Pillar 3 YELLOW-flag          : "
        f"{int(flag_counts.get('YELLOW', 0))} / {len(df)}\n"
        f"  Pillar 3 GREEN-flag           : "
        f"{int(flag_counts.get('GREEN', 0))} / {len(df)}\n"
        f"  RUL: median cycles_to_red     : {red_med:.1f}\n"
        f"  Anomalies flagged             : {n_anom} / {len(df)}\n"
        f"  MC band mean width            : {mean_width:.2f} RCS pts"
    )
    print("\nPipeline finished successfully.")

    return {
        "df": df,
        "scored_df": scored_df,
        "results": results,
        "best_model": best.name,
        "trajectory": trajectory,
        "rul_df": rul_df,
        "anomaly_df": anomaly_df,
    }


if __name__ == "__main__":
    run_pipeline(_parse_args())
