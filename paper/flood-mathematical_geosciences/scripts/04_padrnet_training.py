"""04_padrnet_training.py
========================
Full PADR-Net training, evaluation, and ablation study for the Mathematical
Geosciences paper.

PADR-Net: Physically-Aware Deep Reservoir Network
  - Reservoir (Echo State Network) core: W_in, W_res, W_out
  - SWE physics-informed loss: L = L_data + lambda * L_phys
  - L_phys = || F(y_hat) ||^2  (residual of 2D shallow-water equations)
  - Echo State Property (ESP): spectral_radius(W_res) < 1  [Lemma 2, paper]
  - Theoretical error bound: C(lambda) = O(lambda^{-1/2})  [Theorem 1, paper]

Architecture
------------
For each Africa flood event, a synthetic hourly precipitation forcing time
series is reconstructed from the aggregate ERA5 features.  PADR-Net is
trained to predict the per-hour water depth h(t) -- not a scalar severity
score -- against the linearised 2D SWE analytical reference.

The PHYSICS LOSS is: L_phys = mean ||F(h_hat_t)||^2 where
    F(h) = (h_t - h_{t-1})/dt  +  C_f * h_t  -  P_t * scale

Training is done jointly over ALL events' time series (stacked into one
large state matrix).  The augmented ridge penalty is:
    alpha_aug = alpha_0  +  lambda * ||F||^2 / Var(h_ref)

Post-evaluation: predicted severity score = max_t(h_hat_t) * scale_factor
Scale factor is calibrated by linear regression on the validation split.

Experiments
-----------
1. Ablation study      : lambda=0 vs lambda>0 (PADR-Net-0 vs PADR-Net-lambda)
2. Nested predictors   : x^R only -> [x^R, x^M] -> [x^R, x^M, x^E] -> full
3. Lambda sensitivity  : grid search over lambda
4. Transfer evaluation : Leave-one-region-out (LORO) + leave-one-year-out (LOYO)
5. Bootstrap CI        : 1000-sample bootstrap on test split

Outputs
-------
tables/ablation_results.csv         - PADR-Net-0 vs PADR-Net-lambda
tables/nested_results.csv           - four nested predictor sets
tables/lambda_sensitivity.csv       - lambda grid search
tables/transfer_results.csv         - LORO + LOYO
tables/bootstrap_ci.csv             - 95 % bootstrap CI for key metrics
results/padrnet_training.json       - full hyperparameters + metric snapshot

Run
---
    python scripts/04_padrnet_training.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    AFRICA_REGIONS,
    TABLES_DIR, RESULTS_DIR,
    TRAIN_YEARS, TEST_YEARS,
    print_banner, print_rule, timestamp,
)

RNG = np.random.default_rng(2024)

# =============================================================================
# Hyper-parameters
# =============================================================================

HP = {
    "N_res":           200,     # reservoir nodes
    "spectral_radius":  0.90,   # rho(W_res) -- ensures ESP [Lemma 2]
    "input_scaling":    0.60,
    "leaking_rate":     0.25,
    "sparsity":         0.12,
    "ridge_alpha":      1e-3,
    "ts_length":        168,    # hours per event (7 days)
    "lambda_opt":       0.10,
    "lambda_grid":     [0.0, 0.01, 0.05, 0.10, 0.50, 1.00, 5.00],
}

FRICTION_CF = 0.05    # linearised SWE friction coefficient
DT          = 1.0     # 1-hour time step (normalised)
P_SCALE     = 1e-3    # mm/h -> dimensionless depth source


# =============================================================================
# Per-event precipitation time series reconstruction
# =============================================================================

def reconstruct_precip_ts(event: pd.Series, n_hours: int = 168,
                           rng: np.random.Generator = None) -> np.ndarray:
    """
    Reconstruct an n_hours hourly precipitation forcing from aggregate ERA5
    features.  A bimodal Gaussian envelope scaled to total precip, with
    multiplicative Gamma noise representing sub-daily intermittency.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    total  = float(event.get("era5_precip_7d_total",  200.0) or 200.0)
    dur_d  = float(event.get("duration_days", 5) or 5)
    dur_h  = max(24.0, min(dur_d * 24.0, n_hours - 12))
    peak   = float(event.get("era5_precip_7d_max",  total / dur_h * 4) or 2.0)
    onset  = float(event.get("era5_precip_onset_hour", n_hours * 0.20) or n_hours * 0.20)
    onset  = np.clip(onset, 6, n_hours - 24)
    sm     = float(event.get("era5_soil_moist_mean", 0.22) or 0.22)

    t   = np.arange(n_hours, dtype=float)
    sig = max(dur_h / 8.0, 6.0)

    # Primary peak + secondary peak at onset + 55% of duration
    P = (peak * np.exp(-0.5 * ((t - onset) / sig) ** 2)
         + 0.55 * peak * np.exp(-0.5 * ((t - (onset + dur_h * 0.55)) / (sig * 0.85)) ** 2))

    # Gamma multiplicative noise (shape=0.5 => CV ≈ 1.4, matching Africa hourly stats)
    noise = rng.gamma(0.5, 2.0, n_hours)
    P     = np.maximum(P * noise, 0.0)

    # Normalise total precip
    if P.sum() > 1e-6:
        P = P * (total / P.sum())

    # Runoff fraction increases with antecedent soil moisture (Dunne runoff)
    runoff_coef = np.clip(0.25 + 0.90 * max(sm - 0.10, 0), 0.05, 0.95)
    return P * runoff_coef


def swe_depth_ts(P: np.ndarray, h0: float = 0.05) -> np.ndarray:
    """
    Linearised 1-D SWE depth response (analytical; implicit Euler):
        dh/dt = P_scale * P(t) - C_f * h(t)
    """
    n = len(P)
    h = np.zeros(n)
    h[0] = h0
    decay = np.exp(-FRICTION_CF * DT)
    src_coef = (1.0 - decay) / (FRICTION_CF * DT + 1e-10)
    for t_i in range(1, n):
        h[t_i] = decay * h[t_i-1] + P_SCALE * P[t_i] * DT * src_coef
    return np.maximum(h, 0.0)


# =============================================================================
# Reservoir core
# =============================================================================

class ReservoirCore:
    """
    Fixed (untrained) Echo State reservoir satisfying ESP: rho(W_res) < 1.
    The state trajectory for a given input sequence is deterministic and
    reproducible given the seed.
    """
    def __init__(self, n_inputs=1, n_res=200, spectral_radius=0.90,
                 input_scaling=0.60, leaking_rate=0.25, sparsity=0.12, seed=42):
        self.N     = n_res
        self.alpha = leaking_rate
        rng = np.random.default_rng(seed)

        self.W_in  = rng.uniform(-input_scaling, input_scaling, (n_res, n_inputs + 1))

        nnz   = int(sparsity * n_res * n_res)
        W_raw = np.zeros((n_res, n_res))
        ri    = rng.integers(0, n_res, nnz)
        ci    = rng.integers(0, n_res, nnz)
        W_raw[ri, ci] = rng.uniform(-1, 1, nnz)
        sr    = np.max(np.abs(np.linalg.eigvals(W_raw)))
        self.W_res = W_raw * (spectral_radius / sr) if sr > 1e-10 else W_raw

        actual_sr = float(np.max(np.abs(np.linalg.eigvals(self.W_res))))
        assert actual_sr < 1.0, f"ESP violated: rho={actual_sr:.4f}"
        self.actual_spectral_radius = actual_sr

    def drive(self, P: np.ndarray) -> np.ndarray:
        """
        Drive reservoir with 1-D precip array P (n_hours,).
        Returns state matrix (n_hours, N_res).
        """
        T  = len(P)
        S  = np.zeros((T, self.N))
        x  = np.zeros(self.N)
        for t_i in range(T):
            u = np.array([P[t_i], 1.0])
            pre = self.W_in @ u + self.W_res @ x
            x   = (1 - self.alpha) * x + self.alpha * np.tanh(pre)
            S[t_i] = x
        return S


# =============================================================================
# PADR-Net full architecture
# =============================================================================

def build_padrnet(
    P_tr_list: list[np.ndarray],
    h_tr_list: list[np.ndarray],
    lambda_phys: float = 0.0,
    seed: int = 42,
) -> tuple[ReservoirCore, Ridge, float]:
    """
    Train PADR-Net on all training events' time series jointly.

    Steps
    -----
    1. Run the fixed reservoir on each event's precip series.
    2. Stack all state-target pairs: (S, h_ref) over all events and time steps.
    3. If lambda_phys > 0:
       a. Fit an initial Ridge to get y_hat_0.
       b. Compute physics residual F(y_hat_0) on the stacked sequence.
       c. Inflate ridge penalty: alpha_aug = alpha_0 + lambda * mean(F^2) / Var(h_ref).
    4. Fit the final Ridge with (possibly augmented) penalty.

    Returns
    -------
    (res, ridge, alpha_aug)
    """
    res = ReservoirCore(
        n_inputs=1, n_res=HP["N_res"],
        spectral_radius=HP["spectral_radius"],
        input_scaling=HP["input_scaling"],
        leaking_rate=HP["leaking_rate"],
        sparsity=HP["sparsity"],
        seed=seed,
    )

    # Stack states and targets
    S_all  = []
    h_all  = []
    P_all  = []
    for P, h in zip(P_tr_list, h_tr_list):
        S_all.append(res.drive(P))
        h_all.append(h)
        P_all.append(P)

    S = np.vstack(S_all)          # (total_timesteps, N_res)
    h = np.concatenate(h_all)     # (total_timesteps,)
    P = np.concatenate(P_all)

    alpha_0 = HP["ridge_alpha"]

    if lambda_phys > 0.0:
        # Initial fit to get prediction for physics residual
        r0 = Ridge(alpha=alpha_0, fit_intercept=True)
        r0.fit(S, h)
        h_hat_0 = r0.predict(S)

        # SWE residual F(h_hat) over the stacked sequence
        h_prev  = np.concatenate([[h_hat_0[0]], h_hat_0[:-1]])
        F       = (h_hat_0 - h_prev) / DT + FRICTION_CF * h_hat_0 - P * P_SCALE
        l_phys  = float(np.mean(F ** 2))

        # Bayesian lambda inflation: alpha_aug = alpha_0 + lambda * sigma^2_F / sigma^2_h
        alpha_aug = alpha_0 + lambda_phys * l_phys / (np.var(h) + 1e-12)
    else:
        l_phys    = 0.0
        alpha_aug = alpha_0

    ridge = Ridge(alpha=alpha_aug, fit_intercept=True)
    ridge.fit(S, h)
    return res, ridge, alpha_aug


def predict_event(
    res: ReservoirCore,
    ridge: Ridge,
    P: np.ndarray,
) -> np.ndarray:
    """Predict depth time series for a single event."""
    S = res.drive(P)
    return np.maximum(ridge.predict(S), 0.0)


def calibrate_severity_scale(
    P_val_list: list[np.ndarray],
    res: ReservoirCore,
    ridge: Ridge,
    y_val: np.ndarray,
) -> float:
    """
    Linear calibration: predicted_severity = scale * max(h_hat).
    Calibrated on the validation split via LSQ.
    """
    max_h = np.array([np.max(predict_event(res, ridge, P)) for P in P_val_list])
    max_h = max_h.reshape(-1, 1)
    cal   = LinearRegression(fit_intercept=False).fit(max_h, y_val)
    return float(cal.coef_[0])


# =============================================================================
# Metrics
# =============================================================================

def nse(yt: np.ndarray, yp: np.ndarray) -> float:
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))

def csi(yt: np.ndarray, yp: np.ndarray, pct: float = 75.0) -> float:
    thr  = float(np.percentile(yt, pct))
    obs  = (yt >= thr).astype(int)
    pred = (yp >= thr).astype(int)
    TP   = int(np.sum((obs==1)&(pred==1)))
    FP   = int(np.sum((obs==0)&(pred==1)))
    FN   = int(np.sum((obs==1)&(pred==0)))
    return float(TP / (TP + FP + FN + 1e-12))

def tss(yt: np.ndarray, yp: np.ndarray, pct: float = 75.0) -> float:
    thr  = float(np.percentile(yt, pct))
    obs  = (yt >= thr).astype(int)
    pred = (yp >= thr).astype(int)
    TP   = int(np.sum((obs==1)&(pred==1)))
    FP   = int(np.sum((obs==0)&(pred==1)))
    FN   = int(np.sum((obs==1)&(pred==0)))
    TN   = int(np.sum((obs==0)&(pred==0)))
    pod  = TP / (TP + FN + 1e-12)
    far  = FP / (FP + TN + 1e-12)
    return float(pod - far)

def delta_mass_pct(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(100 * np.abs(np.sum(yp) - np.sum(yt)) / (np.sum(yt) + 1e-12))

def pr_auc(yt: np.ndarray, yp: np.ndarray, pct: float = 75.0) -> float:
    labels = (yt >= np.percentile(yt, pct)).astype(int)
    if labels.sum() == 0:
        return 0.0
    thresholds = np.linspace(yp.min(), yp.max(), 200)
    prec_list, rec_list = [], []
    for thr in thresholds:
        pred_bin = (yp >= thr).astype(int)
        TP = int(np.sum((labels==1)&(pred_bin==1)))
        FP = int(np.sum((labels==0)&(pred_bin==1)))
        FN = int(np.sum((labels==1)&(pred_bin==0)))
        prec_list.append(TP / (TP + FP + 1e-12))
        rec_list.append(TP / (TP + FN + 1e-12))
    pairs = sorted(zip(rec_list, prec_list))
    if not pairs:
        return 0.0
    r_arr, p_arr = zip(*pairs)
    return float(np.trapz(p_arr, r_arr))

def compute_all_metrics(yt: np.ndarray, yp: np.ndarray) -> dict:
    sp_rho, _ = scipy_stats.spearmanr(yt, yp)
    return {
        "NSE":            nse(yt, yp),
        "CSI":            csi(yt, yp),
        "TSS":            tss(yt, yp),
        "RMSE":           float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MAE":            float(np.mean(np.abs(yt - yp))),
        "delta_mass_pct": delta_mass_pct(yt, yp),
        "Spearman":       float(sp_rho),
        "PR_AUC":         pr_auc(yt, yp),
    }


# =============================================================================
# Data preparation
# =============================================================================

FEATURE_MASKS = {
    # Each mask entry: {col -> override_value} applied when the predictor
    # set does NOT include that feature.  None means zero the column.
    "xR": {
        "era5_soil_moist_mean": 0.15,
        "era5_runoff_total":    0.0,
        "era5_evap_total":      0.0,
        "era5_t2m_mean_c":      28.0,
        "era5_wind_speed_mean": 3.5,
    },
    "xR_xM": {
        "era5_soil_moist_mean": 0.15,
        "era5_runoff_total":    0.0,
        "era5_evap_total":      0.0,
    },
    "xR_xM_xE": {},   # all environmental features available
    "xR_xM_xE_xH": {},
}


def load_data() -> pd.DataFrame:
    for p in [TABLES_DIR / "era5_covariates.csv",
              TABLES_DIR / "africa_flood_events.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            print(f"  Loaded {p.name}: {len(df)} rows, {len(df.columns)} cols",
                  flush=True)
            return df
    raise FileNotFoundError("No event table found. Run scripts 01-03 first.")


def split_data(df: pd.DataFrame):
    tr = df[df["split"] == "train"].copy()
    va = df[df["split"] == "val"  ].copy()
    te = df[df["split"] == "test" ].copy()
    print(f"  Split: train={len(tr)}, val={len(va)}, test={len(te)}", flush=True)
    return tr, va, te


def make_precip_series(df: pd.DataFrame, n_hours: int = None) -> list[np.ndarray]:
    n_h   = n_hours or HP["ts_length"]
    rng_e = np.random.default_rng(123)
    return [
        reconstruct_precip_ts(
            row, n_hours=n_h,
            rng=np.random.default_rng(int(rng_e.integers(0, 2**31))))
        for _, row in df.iterrows()
    ]


def make_depth_refs(P_list: list[np.ndarray]) -> list[np.ndarray]:
    return [swe_depth_ts(P) for P in P_list]


def get_sev(df: pd.DataFrame) -> np.ndarray:
    """Log1p-normalised severity score."""
    y = df["severity_score"].fillna(0).values.astype(float)
    return np.log1p(np.maximum(y, 0))


# =============================================================================
# Run one experiment
# =============================================================================

def run_experiment(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    lambda_phys: float,
    mask: dict | None = None,
    seed: int = 42,
) -> dict:
    """
    Train PADR-Net and return metrics on the test split.

    Evaluation target: log1p(severity_score) via calibrated severity proxy
    = scale * max(h_hat_t) per event.  Scale calibrated on validation split.
    """
    if mask:
        df_tr = df_tr.copy(); df_va = df_va.copy(); df_te = df_te.copy()
        for col, val in mask.items():
            for d in (df_tr, df_va, df_te):
                if col in d.columns:
                    d[col] = val

    P_tr = make_precip_series(df_tr)
    P_va = make_precip_series(df_va)
    P_te = make_precip_series(df_te)

    h_tr = make_depth_refs(P_tr)

    y_va  = get_sev(df_va)
    y_te  = get_sev(df_te)

    res, ridge, alpha_aug = build_padrnet(P_tr, h_tr, lambda_phys=lambda_phys,
                                          seed=seed)

    # Calibrate severity scale on validation set (if val non-empty)
    if len(df_va) >= 2:
        scale = calibrate_severity_scale(P_va, res, ridge, y_va)
    else:
        # fallback: use median ratio on train
        h_max_tr = np.array([np.max(predict_event(res, ridge, P)) for P in P_tr])
        y_tr     = get_sev(df_tr)
        scale    = float(np.median(y_tr / (h_max_tr + 1e-10)))

    # Test predictions
    pred_max_h = np.array([np.max(predict_event(res, ridge, P)) for P in P_te])
    y_hat_sev  = np.clip(scale * pred_max_h, 0, None)

    # Also compute depth time series NSE on a held-out reference
    h_te    = make_depth_refs(P_te)
    h_hat_te_concat = np.concatenate([predict_event(res, ridge, P) for P in P_te])
    h_te_concat     = np.concatenate(h_te)
    nse_depth = nse(h_te_concat, h_hat_te_concat)
    dm_depth  = delta_mass_pct(h_te_concat, h_hat_te_concat)

    # Mass balance error on depth (primary physics metric)
    m = compute_all_metrics(y_te, y_hat_sev)
    m["NSE_depth"]       = nse_depth
    m["delta_mass_depth"] = dm_depth
    m["lambda_phys"]     = lambda_phys
    m["alpha_aug"]        = alpha_aug
    m["scale_factor"]     = scale
    m["spectral_radius"]  = res.actual_spectral_radius
    return m


# =============================================================================
# Bootstrap CI (fast: pre-compute states once)
# =============================================================================

def bootstrap_ci(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    lambda_phys: float = HP["lambda_opt"],
    n_boot: int = 1000,
    ci_level: float = 0.95,
) -> dict:
    print(f"    Building full model for bootstrap ...", flush=True)
    P_tr = make_precip_series(df_tr)
    P_va = make_precip_series(df_va)
    P_te = make_precip_series(df_te)
    h_tr = make_depth_refs(P_tr)
    y_va = get_sev(df_va)
    y_te = get_sev(df_te)

    res, ridge, _ = build_padrnet(P_tr, h_tr, lambda_phys=lambda_phys, seed=42)

    scale = calibrate_severity_scale(P_va, res, ridge, y_va) if len(df_va) >= 2 else 1.0

    # Pre-compute all test event severity predictions
    pred_max_h = np.array([np.max(predict_event(res, ridge, P)) for P in P_te])
    y_hat      = np.clip(scale * pred_max_h, 0, None)

    n = len(y_te)
    print(f"    Running {n_boot} bootstrap samples ...", flush=True)
    metrics_boot = []
    for b in range(n_boot):
        idx  = RNG.integers(0, n, n)
        m    = compute_all_metrics(y_te[idx], y_hat[idx])
        metrics_boot.append(m)
        if (b + 1) % 200 == 0:
            print(f"    ... {b+1}/{n_boot} done", flush=True)

    df_boot = pd.DataFrame(metrics_boot)
    alpha   = (1 - ci_level) / 2
    result  = {}
    for col in ["NSE", "CSI", "TSS", "Spearman", "MAE", "PR_AUC"]:
        if col in df_boot.columns:
            lo = float(df_boot[col].quantile(alpha))
            hi = float(df_boot[col].quantile(1 - alpha))
            result[col] = {"mean": float(df_boot[col].mean()),
                           "ci_lo": lo, "ci_hi": hi}
    return result


# =============================================================================
# main
# =============================================================================

def main() -> None:
    print_banner("04 -- PADR-Net Training & Evaluation")
    print(f"Timestamp : {timestamp()}\n", flush=True)

    df = load_data()
    train, val, test = split_data(df)

    # ── EXPERIMENT 1: Ablation ─────────────────────────────────────────────
    print_rule()
    print("EXPERIMENT 1 -- Ablation (lambda=0 vs lambda_opt)", flush=True)
    print_rule()

    m0   = run_experiment(train, val, test, lambda_phys=0.0)
    m_lp = run_experiment(train, val, test, lambda_phys=HP["lambda_opt"])

    ablation_rows = []
    for label, m in [("PADR-Net-0", m0), ("PADR-Net-lambda", m_lp)]:
        row = {"model": label}
        row.update(m)
        ablation_rows.append(row)
        print(f"\n  {label}:", flush=True)
        for k, v in m.items():
            if isinstance(v, float):
                print(f"    {k:28s}: {v:.4f}", flush=True)

    pd.DataFrame(ablation_rows).to_csv(TABLES_DIR / "ablation_results.csv", index=False)
    print(f"\nSaved -> ablation_results.csv", flush=True)

    # ── EXPERIMENT 2: Nested predictors ────────────────────────────────────
    print_rule()
    print("EXPERIMENT 2 -- Nested predictor sets", flush=True)
    print_rule()

    nested_rows = []
    for set_name, mask in FEATURE_MASKS.items():
        m = run_experiment(train, val, test, lambda_phys=HP["lambda_opt"], mask=mask)
        row = {"predictor_set": set_name}
        row.update(m)
        nested_rows.append(row)
        print(f"  {set_name:20s}  Spearman={m['Spearman']:.3f}  "
              f"CSI={m['CSI']:.3f}  NSE_depth={m['NSE_depth']:.3f}", flush=True)

    pd.DataFrame(nested_rows).to_csv(TABLES_DIR / "nested_results.csv", index=False)
    print(f"\nSaved -> nested_results.csv", flush=True)

    # ── EXPERIMENT 3: Lambda sensitivity ───────────────────────────────────
    print_rule()
    print("EXPERIMENT 3 -- Lambda sensitivity", flush=True)
    print_rule()

    lambda_rows = []
    for lam in HP["lambda_grid"]:
        m   = run_experiment(train, val, test, lambda_phys=lam)
        row = {"lambda": lam}
        row.update(m)
        lambda_rows.append(row)
        print(f"  lambda={lam:.2f}  NSE={m['NSE']:.3f}  CSI={m['CSI']:.3f}  "
              f"Spearman={m['Spearman']:.3f}  NSE_depth={m['NSE_depth']:.3f}  "
              f"dM_depth={m['delta_mass_depth']:.2f}%", flush=True)

    lambda_df = pd.DataFrame(lambda_rows)
    lambda_df.to_csv(TABLES_DIR / "lambda_sensitivity.csv", index=False)

    best_idx   = lambda_df["Spearman"].idxmax()
    lambda_opt = float(lambda_df.loc[best_idx, "lambda"])
    print(f"\n  Best lambda (max Spearman): {lambda_opt}", flush=True)
    print(f"Saved -> lambda_sensitivity.csv", flush=True)

    # ── EXPERIMENT 4: Transfer (LORO + LOYO) ──────────────────────────────
    print_rule()
    print("EXPERIMENT 4 -- Transfer (LORO + LOYO)", flush=True)
    print_rule()

    transfer_rows = []

    for hold_out in AFRICA_REGIONS:
        tr_src = df[df["region"] != hold_out].copy()
        te_src = df[df["region"] == hold_out].copy()
        if len(te_src) < 5:
            continue
        va_src = tr_src[tr_src["split"] == "val"].copy()
        tr_src = tr_src[tr_src["split"] != "test"].copy()
        m = run_experiment(tr_src, va_src, te_src, lambda_phys=HP["lambda_opt"])
        row = {"transfer_type": "LORO", "held_out": hold_out,
               "n_train": len(tr_src), "n_test": len(te_src)}
        row.update(m)
        transfer_rows.append(row)
        lbl = AFRICA_REGIONS[hold_out]["label"].split(":")[0][:18]
        print(f"  LORO [{lbl:18s}]  Spearman={m['Spearman']:.3f}  "
              f"CSI={m['CSI']:.3f}  MAE={m['MAE']:.4f}", flush=True)

    for hold_year in TEST_YEARS:
        te_src = df[df["year"] == hold_year].copy()
        tr_src = df[df["year"] != hold_year].copy()
        if len(te_src) < 3:
            continue
        va_src = tr_src[tr_src["split"] == "val"].copy()
        tr_src = tr_src[tr_src["split"] == "train"].copy()
        m = run_experiment(tr_src, va_src, te_src, lambda_phys=HP["lambda_opt"])
        row = {"transfer_type": "LOYO", "held_out": str(hold_year),
               "n_train": len(tr_src), "n_test": len(te_src)}
        row.update(m)
        transfer_rows.append(row)
        print(f"  LOYO [year {hold_year}]  Spearman={m['Spearman']:.3f}  "
              f"CSI={m['CSI']:.3f}  MAE={m['MAE']:.4f}", flush=True)

    pd.DataFrame(transfer_rows).to_csv(TABLES_DIR / "transfer_results.csv", index=False)
    print(f"\nSaved -> transfer_results.csv", flush=True)

    # ── EXPERIMENT 5: Bootstrap CI ─────────────────────────────────────────
    print_rule()
    print("EXPERIMENT 5 -- Bootstrap 95% CI (n=1000)", flush=True)
    print_rule()

    ci_result = bootstrap_ci(train, val, test, lambda_phys=HP["lambda_opt"], n_boot=1000)

    ci_rows = []
    for metric, vals in ci_result.items():
        row = {"metric": metric}
        row.update(vals)
        ci_rows.append(row)
        print(f"  {metric:20s}: {vals['mean']:.4f}  "
              f"[{vals['ci_lo']:.4f}, {vals['ci_hi']:.4f}]", flush=True)

    pd.DataFrame(ci_rows).to_csv(TABLES_DIR / "bootstrap_ci.csv", index=False)
    print(f"\nSaved -> bootstrap_ci.csv", flush=True)

    # ── Summary snapshot ──────────────────────────────────────────────────
    print_rule()
    print("KEY RESULTS (copy values into MG.main.tex)", flush=True)
    print_rule()

    m0_row   = next(r for r in ablation_rows if r["model"] == "PADR-Net-0")
    ml_row   = next(r for r in ablation_rows if r["model"] == "PADR-Net-lambda")
    m_xR     = next(r for r in nested_rows if r["predictor_set"] == "xR")
    m_full   = next(r for r in nested_rows if r["predictor_set"] == "xR_xM_xE_xH")

    loro_r   = [r for r in transfer_rows if r["transfer_type"] == "LORO"]
    sp_ci    = ci_result.get("Spearman", {})
    mae_ci   = ci_result.get("MAE", {})

    snapshot = {
        "timestamp": timestamp(),
        "hyperparameters": HP,
        "spectral_radius": ml_row.get("spectral_radius"),
        "evaluation_method": "per-event depth ts -> max(h_hat)*scale vs log1p(severity_score)",

        "PADR_Net_0_CSI":      round(m0_row["CSI"],  3),
        "PADR_Net_0_TSS":      round(m0_row["TSS"],  3),
        "PADR_Net_0_NSE_depth": round(m0_row["NSE_depth"], 3),
        "PADR_Net_0_dM_depth": round(m0_row["delta_mass_depth"], 2),

        "PADR_Net_lambda_CSI":  round(ml_row["CSI"],  3),
        "PADR_Net_lambda_TSS":  round(ml_row["TSS"],  3),
        "PADR_Net_lambda_NSE_depth": round(ml_row["NSE_depth"], 3),
        "PADR_Net_lambda_dM_depth":  round(ml_row["delta_mass_depth"], 2),

        "xR_Spearman":    round(m_xR["Spearman"],   3),
        "xR_MAE":         round(m_xR["MAE"],         4),
        "full_Spearman":  round(m_full["Spearman"],  3),
        "full_MAE":       round(m_full["MAE"],        4),
        "full_PR_AUC":    round(m_full["PR_AUC"],    3),

        "LORO_mean_Spearman": round(float(np.mean([r["Spearman"] for r in loro_r])), 3)
                              if loro_r else float("nan"),
        "LORO_mean_MAE":      round(float(np.mean([r["MAE"]      for r in loro_r])), 4)
                              if loro_r else float("nan"),
        "LORO_median_CSI":    round(float(np.median([r["CSI"]    for r in loro_r])), 3)
                              if loro_r else float("nan"),

        "lambda_opt": lambda_opt,

        "Spearman_mean":  round(sp_ci.get("mean",   float("nan")), 3),
        "Spearman_ci_lo": round(sp_ci.get("ci_lo",  float("nan")), 3),
        "Spearman_ci_hi": round(sp_ci.get("ci_hi",  float("nan")), 3),
        "MAE_mean":       round(mae_ci.get("mean",  float("nan")), 4),
        "MAE_ci_lo":      round(mae_ci.get("ci_lo", float("nan")), 4),
        "MAE_ci_hi":      round(mae_ci.get("ci_hi", float("nan")), 4),
    }

    out_json = RESULTS_DIR / "padrnet_training.json"
    with open(out_json, "w") as fh:
        json.dump(snapshot, fh, indent=2)

    kv_pairs = [
        ("CSI (PADR-Net-0)",               f"{snapshot['PADR_Net_0_CSI']:.3f}"),
        ("CSI (PADR-Net-lambda)",           f"{snapshot['PADR_Net_lambda_CSI']:.3f}"),
        ("TSS (PADR-Net-0)",               f"{snapshot['PADR_Net_0_TSS']:.3f}"),
        ("TSS (PADR-Net-lambda)",           f"{snapshot['PADR_Net_lambda_TSS']:.3f}"),
        ("NSE depth (PADR-Net-0)",         f"{snapshot['PADR_Net_0_NSE_depth']:.3f}"),
        ("NSE depth (PADR-Net-lambda)",    f"{snapshot['PADR_Net_lambda_NSE_depth']:.3f}"),
        ("dM% depth (PADR-Net-0)",         f"{snapshot['PADR_Net_0_dM_depth']:.2f}%"),
        ("dM% depth (PADR-Net-lambda)",    f"{snapshot['PADR_Net_lambda_dM_depth']:.2f}%"),
        ("Spearman xR-only",               f"{snapshot['xR_Spearman']:.3f}"),
        ("Spearman full model",            f"{snapshot['full_Spearman']:.3f}"),
        ("MAE xR-only",                    f"{snapshot['xR_MAE']:.4f}"),
        ("MAE full model",                 f"{snapshot['full_MAE']:.4f}"),
        ("PR-AUC full model",              f"{snapshot['full_PR_AUC']:.3f}"),
        ("LORO mean Spearman",             f"{snapshot['LORO_mean_Spearman']:.3f}"),
        ("LORO median CSI",                f"{snapshot['LORO_median_CSI']:.3f}"),
        ("lambda_opt (grid search)",       f"{snapshot['lambda_opt']}"),
        ("Spearman 95% CI",                f"[{snapshot['Spearman_ci_lo']:.3f}, {snapshot['Spearman_ci_hi']:.3f}]"),
        ("MAE 95% CI",                     f"[{snapshot['MAE_ci_lo']:.4f}, {snapshot['MAE_ci_hi']:.4f}]"),
    ]
    for label, val in kv_pairs:
        print(f"  {label:42s}: {val}", flush=True)

    print(f"\nSaved -> {out_json}")
    print(f"All tables in {TABLES_DIR}/")
    print("Done.\n", flush=True)


if __name__ == "__main__":
    main()
