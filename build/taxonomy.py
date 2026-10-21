"""
02_taxonomy.py — Class taxonomy + statistical verification battery.

This module is the SINGLE SOURCE OF TRUTH for labels. Both the real-mining track
(03) and the synthesis track (04) call the exact same verify_* functions, so a
label exists only when a named statistical test confirms it on the *precise
array that will be plotted* — never on a parent dataframe, never by assumption.

Exclusivity: every verify_* returns a label only if exactly one class's guard set
passes. Multi-match candidates are rejected as ambiguous. This keeps the primary
label single-label and clean for a confusion matrix.
"""
import warnings

import numpy as np
from scipy import signal, stats

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Class registry
# --------------------------------------------------------------------------
CLASSES = {
    # ---- Family A: univariate distribution shape -------------------------
    "dist_normal": dict(
        family="distribution", arity="univariate",
        charts=["histogram", "kde", "boxplot", "violin", "ecdf", "qq", "boxen", "strip"],
        test="Shapiro-Wilk + moment bounds + Hartigan dip",
        desc="Approximately Gaussian: symmetric, unimodal, mesokurtic, no outlier mass."),
    "dist_right_skew": dict(
        family="distribution", arity="univariate",
        charts=["histogram", "kde", "boxplot", "violin", "ecdf", "qq", "boxen", "strip"],
        test="Fisher-Pearson skewness + Shapiro-Wilk",
        desc="Long right tail; mean exceeds median."),
    "dist_left_skew": dict(
        family="distribution", arity="univariate",
        charts=["histogram", "kde", "boxplot", "violin", "ecdf", "qq", "boxen", "strip"],
        test="Fisher-Pearson skewness + Shapiro-Wilk",
        desc="Long left tail; mean below median."),
    "dist_bimodal": dict(
        family="distribution", arity="univariate",
        charts=["histogram", "kde", "violin", "ecdf", "strip", "boxen"],
        test="Hartigan dip test + 2-component GMM BIC",
        desc="Two separated modes; a single central tendency is misleading."),
    "dist_heavy_tail": dict(
        family="distribution", arity="univariate",
        charts=["histogram", "kde", "boxplot", "violin", "qq", "boxen", "strip"],
        test="Excess kurtosis + IQR outlier fraction",
        desc="Leptokurtic with extreme values far beyond the whiskers."),
    "dist_uniform": dict(
        family="distribution", arity="univariate",
        charts=["histogram", "kde", "ecdf", "boxplot", "strip", "boxen"],
        test="KS test against Uniform + platykurtic bound",
        desc="Flat density; no preferred region across the range."),

    # ---- Family B: bivariate relationship --------------------------------
    "rel_strong_positive": dict(
        family="relationship", arity="bivariate",
        charts=["scatter", "hexbin", "jointplot", "bubble", "regplot"],
        test="Pearson r with Spearman agreement",
        desc="Strong positive linear association."),
    "rel_strong_negative": dict(
        family="relationship", arity="bivariate",
        charts=["scatter", "hexbin", "jointplot", "bubble", "regplot"],
        test="Pearson r with Spearman agreement",
        desc="Strong negative linear association."),
    "rel_none": dict(
        family="relationship", arity="bivariate",
        charts=["scatter", "hexbin", "jointplot", "bubble", "regplot"],
        test="Pearson + Spearman + quadratic R^2 all near zero",
        desc="No detectable association of any modelled form."),
    "rel_nonlinear_monotonic": dict(
        family="relationship", arity="bivariate",
        charts=["scatter", "hexbin", "jointplot", "bubble", "regplot"],
        test="Spearman >> Pearson divergence",
        desc="Monotonic but curved; Pearson understates the true dependence."),
    "rel_nonlinear_nonmonotonic": dict(
        family="relationship", arity="bivariate",
        charts=["scatter", "hexbin", "jointplot", "bubble", "regplot"],
        test="Quadratic R^2 with significant curvature, linear r ~ 0",
        desc="U- or inverted-U shape; linear correlation is blind to it."),
    "rel_heteroscedastic": dict(
        family="relationship", arity="bivariate",
        charts=["scatter", "hexbin", "jointplot", "regplot"],
        test="Breusch-Pagan + residual-spread ratio across x terciles",
        desc="Variance of y grows with x; a fan/cone shape."),

    # ---- Family C: temporal ----------------------------------------------
    "ts_increasing": dict(
        family="temporal", arity="timeseries",
        charts=["line", "area", "step", "line_marker"],
        test="Mann-Kendall tau (non-parametric monotonic trend)",
        desc="Sustained upward trend over the observed window."),
    "ts_decreasing": dict(
        family="temporal", arity="timeseries",
        charts=["line", "area", "step", "line_marker"],
        test="Mann-Kendall tau (non-parametric monotonic trend)",
        desc="Sustained downward trend over the observed window."),
    "ts_seasonal": dict(
        family="temporal", arity="timeseries",
        charts=["line", "area", "step", "line_marker"],
        test="Periodogram dominant power + ACF at the implied lag",
        desc="Repeating cycle at a fixed period; no net trend."),
    "ts_changepoint": dict(
        family="temporal", arity="timeseries",
        charts=["line", "area", "step", "line_marker"],
        test="Binary-segmentation level shift vs residual scale",
        desc="Abrupt regime shift in level partway through the series."),
    "ts_anomaly_spike": dict(
        family="temporal", arity="timeseries",
        charts=["line", "line_marker", "step"],
        test="Median/MAD robust z-score point anomalies",
        desc="Otherwise stable series with isolated extreme point(s)."),

    # ---- Family D: group comparison / composition ------------------------
    "grp_significant_diff": dict(
        family="group", arity="grouped",
        charts=["grouped_box", "grouped_violin", "grouped_bar", "pointplot", "grouped_strip"],
        test="Kruskal-Wallis H + epsilon-squared effect size",
        desc="Groups differ substantially in central tendency (large effect)."),
    "grp_no_diff": dict(
        family="group", arity="grouped",
        charts=["grouped_box", "grouped_violin", "grouped_bar", "pointplot", "grouped_strip"],
        test="Kruskal-Wallis H + epsilon-squared effect size",
        desc="Groups are statistically indistinguishable; apparent gaps are noise."),
    "grp_dominant_category": dict(
        family="group", arity="composition",
        charts=["countplot", "pie", "barh", "bar"],
        test="Top-category share + normalised Shannon entropy",
        desc="One category dominates the composition."),
}

CLASS_NAMES = list(CLASSES)
FAMILIES = sorted({c["family"] for c in CLASSES.values()})


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _f(v):
    """JSON-safe float."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else round(v, 6)


def _clean(a):
    a = np.asarray(a, dtype=float).ravel()
    return a[np.isfinite(a)]


def _resolve(matches):
    """Single-label discipline: exactly one match, else reject."""
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, "no_class_matched"
    return None, "ambiguous:" + "+".join(matches)


def _dip(x):
    try:
        import diptest
        d, p = diptest.diptest(np.ascontiguousarray(x, dtype=float))
        return _f(d), _f(p)
    except Exception:
        return None, None


def _gmm_bic(x):
    """BIC for 1 vs 2 components, plus mode separation in pooled-SD units."""
    try:
        from sklearn.mixture import GaussianMixture
        z = ((x - x.mean()) / (x.std() + 1e-12)).reshape(-1, 1)
        if len(z) > 3000:
            z = z[np.random.default_rng(0).choice(len(z), 3000, replace=False)]
        g1 = GaussianMixture(1, random_state=0, n_init=2).fit(z)
        g2 = GaussianMixture(2, random_state=0, n_init=3).fit(z)
        mu = np.sort(g2.means_.ravel())
        sd = np.sqrt(g2.covariances_.ravel())
        pooled = float(np.sqrt(np.mean(sd ** 2)))
        sep = float(abs(mu[1] - mu[0]) / (pooled + 1e-12))
        w = float(g2.weights_.min())
        return _f(g1.bic(z)), _f(g2.bic(z)), _f(sep), _f(w)
    except Exception:
        return None, None, None, None


# ==========================================================================
# FAMILY A — univariate distribution
# ==========================================================================
MIN_N_UNI = 60


def verify_univariate(x):
    x = _clean(x)
    n = len(x)
    if n < MIN_N_UNI:
        return None, {}, "too_few_points"
    if len(np.unique(x)) < 15:
        return None, {}, "too_discrete"
    if np.std(x) <= 0:
        return None, {}, "zero_variance"

    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((x < lo) | (x > hi)).sum())

    sk = float(stats.skew(x, bias=False))
    ku = float(stats.kurtosis(x, fisher=True, bias=False))  # excess
    # Bowley (Yule-Kendall) quartile skewness: bounded in [-1,1] and robust to the
    # tails. Moment skewness is driven by single extreme points, so a symmetric
    # heavy-tailed sample can read |skew| > 1 and be mislabelled as skewed.
    bowley = float((q3 + q1 - 2 * q2) / (iqr + 1e-12)) if iqr > 0 else 0.0

    sub = x if n <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
    try:
        sh_stat, sh_p = stats.shapiro(sub)
    except Exception:
        sh_stat, sh_p = np.nan, np.nan

    dip_stat, dip_p = _dip(x)
    bic1, bic2, gsep, gw = _gmm_bic(x)

    rng = x.max() - x.min()
    u = (x - x.min()) / (rng + 1e-12)
    ks_u_stat, ks_u_p = stats.kstest(u, "uniform")

    try:
        ad = stats.anderson(sub, dist="norm")
        ad_stat, ad_crit5 = float(ad.statistic), float(ad.critical_values[2])
    except Exception:
        ad_stat = ad_crit5 = None

    s = dict(
        n=n, mean=_f(x.mean()), median=_f(q2), std=_f(x.std(ddof=1)),
        min=_f(x.min()), max=_f(x.max()), q1=_f(q1), q3=_f(q3), iqr=_f(iqr),
        skewness=_f(sk), excess_kurtosis=_f(ku), bowley_skew=_f(bowley),
        shapiro_stat=_f(sh_stat), shapiro_p=_f(sh_p),
        anderson_stat=_f(ad_stat), anderson_crit_5pct=_f(ad_crit5),
        dip_stat=dip_stat, dip_p=dip_p,
        gmm_bic_1=bic1, gmm_bic_2=bic2, gmm_mode_separation=gsep, gmm_min_weight=gw,
        ks_uniform_stat=_f(ks_u_stat), ks_uniform_p=_f(ks_u_p),
        outlier_count=n_out, outlier_pct=_f(100.0 * n_out / n),
        iqr_lower_fence=_f(lo), iqr_upper_fence=_f(hi),
        cv=_f(x.std(ddof=1) / abs(x.mean())) if abs(x.mean()) > 1e-9 else None,
    )

    dp = dip_p if dip_p is not None else 1.0
    shp = sh_p if np.isfinite(sh_p) else 1.0
    bic_gain = (bic1 - bic2) if (bic1 is not None and bic2 is not None) else -1e9

    m = []
    if shp > 0.05 and abs(sk) < 0.35 and abs(ku) < 0.6 and dp > 0.15 and s["outlier_pct"] < 3.0:
        m.append("dist_normal")
    # moment skew and robust quartile skew must agree, else the tail is driving it
    if sk > 1.0 and bowley > 0.05 and shp < 0.01 and dp > 0.10:
        m.append("dist_right_skew")
    if sk < -1.0 and bowley < -0.05 and shp < 0.01 and dp > 0.10:
        m.append("dist_left_skew")
    # ks_uniform_p guard is essential: the dip test is calibrated against the
    # uniform, so a genuine uniform sample trips dip_p < 0.05 ~5% of the time and
    # a flat density is also poorly fit by one Gaussian. Measured separation is
    # total (bimodal ks_p ~ 0.0000 vs uniform ks_p > 0.16).
    if (dp < 0.05 and bic_gain > 10 and (gsep or 0) > 1.2 and (gw or 0) > 0.15
            and ks_u_p < 0.05):
        m.append("dist_bimodal")
    # sample excess kurtosis of a heavy-tailed law has very high variance
    # (t(4), n=600: median 4.5 but minimum 2.1 over 12 draws), so the bound is
    # set from the observed lower tail and paired with an outlier-mass criterion
    if ku > 2.0 and s["outlier_pct"] > 2.5 and abs(bowley) < 0.10 and dp > 0.10:
        m.append("dist_heavy_tail")
    if ks_u_p > 0.10 and ku < -0.9 and dp > 0.15 and abs(sk) < 0.3:
        m.append("dist_uniform")

    label, why = _resolve(m)
    return label, s, why


# ==========================================================================
# FAMILY B — bivariate relationship
# ==========================================================================
MIN_N_BI = 60


def verify_bivariate(x, y):
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < MIN_N_BI:
        return None, {}, "too_few_points"
    if len(np.unique(x)) < 15 or len(np.unique(y)) < 15:
        return None, {}, "too_discrete"
    if np.std(x) <= 0 or np.std(y) <= 0:
        return None, {}, "zero_variance"

    r, r_p = stats.pearsonr(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    tau, tau_p = stats.kendalltau(x, y)

    b1, b0 = np.polyfit(x, y, 1)
    lin_pred = b1 * x + b0
    resid = y - lin_pred
    sst = float(((y - y.mean()) ** 2).sum())
    lin_r2 = 1 - float((resid ** 2).sum()) / (sst + 1e-12)

    # quadratic fit on standardised x for conditioning
    xs = (x - x.mean()) / (x.std() + 1e-12)
    c2, c1, c0 = np.polyfit(xs, y, 2)
    quad_pred = c2 * xs ** 2 + c1 * xs + c0
    quad_r2 = 1 - float(((y - quad_pred) ** 2).sum()) / (sst + 1e-12)

    # significance of the curvature term
    X = np.column_stack([np.ones(n), xs, xs ** 2])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res2 = y - X @ beta
        s2 = float((res2 ** 2).sum()) / max(n - 3, 1)
        cov = s2 * np.linalg.inv(X.T @ X)
        t_curv = float(beta[2] / (np.sqrt(cov[2, 2]) + 1e-12))
    except Exception:
        t_curv = 0.0

    # Breusch-Pagan on linear residuals
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(resid, np.column_stack([np.ones(n), x]))
    except Exception:
        bp_lm = bp_p = None

    # residual spread across x terciles (fan detector)
    order = np.argsort(x)
    k = n // 3
    lo_sd = float(np.std(resid[order][:k])) if k > 5 else np.nan
    hi_sd = float(np.std(resid[order][-k:])) if k > 5 else np.nan
    spread_ratio = float(max(hi_sd, lo_sd) / (min(hi_sd, lo_sd) + 1e-12)) if k > 5 else 1.0

    s = dict(
        n=n, pearson_r=_f(r), pearson_p=_f(r_p),
        spearman_rho=_f(rho), spearman_p=_f(rho_p),
        kendall_tau=_f(tau), kendall_p=_f(tau_p),
        linear_slope=_f(b1), linear_intercept=_f(b0), linear_r2=_f(lin_r2),
        quadratic_r2=_f(quad_r2), curvature_coef=_f(c2), curvature_t=_f(t_curv),
        bp_lm_stat=_f(bp_lm), bp_p=_f(bp_p), resid_spread_ratio=_f(spread_ratio),
        x_mean=_f(x.mean()), x_std=_f(x.std(ddof=1)),
        y_mean=_f(y.mean()), y_std=_f(y.std(ddof=1)),
    )

    bpp = bp_p if bp_p is not None else 1.0
    m = []
    if r > 0.70 and r_p < 0.01 and abs(r - rho) < 0.09 and bpp > 0.01:
        m.append("rel_strong_positive")
    if r < -0.70 and r_p < 0.01 and abs(r - rho) < 0.09 and bpp > 0.01:
        m.append("rel_strong_negative")
    if abs(r) < 0.12 and abs(rho) < 0.12 and quad_r2 < 0.05:
        m.append("rel_none")
    # The signal is rank/linear *divergence* plus a poor linear fit, not an
    # absolute Pearson bound: measured over exp(3x), x^6, 1/x the divergence is
    # only 0.11-0.34, so a 0.20 cut misses most genuinely curved monotone data.
    # The 0.09 / 0.10 gap against rel_strong_* keeps the two disjoint.
    if abs(rho) > 0.80 and (abs(rho) - abs(r)) > 0.10 and lin_r2 < 0.72 and rho_p < 0.01:
        m.append("rel_nonlinear_monotonic")
    if abs(r) < 0.25 and abs(rho) < 0.35 and quad_r2 > 0.45 and abs(t_curv) > 4:
        m.append("rel_nonlinear_nonmonotonic")
    if bpp < 0.001 and spread_ratio > 2.5 and quad_r2 < 0.6:
        m.append("rel_heteroscedastic")

    label, why = _resolve(m)
    return label, s, why


# ==========================================================================
# FAMILY C — temporal
# ==========================================================================
MIN_N_TS, MAX_N_TS = 24, 500


def _mk(y):
    try:
        import pymannkendall as mk
        r = mk.original_test(y)
        return _f(r.Tau), _f(r.p), _f(r.slope), str(r.trend)
    except Exception:
        n = len(y)
        t = np.arange(n)
        tau, p = stats.kendalltau(t, y)
        sl = np.polyfit(t, y, 1)[0]
        return _f(tau), _f(p), _f(sl), ("increasing" if tau > 0 else "decreasing")


def _seasonality(y):
    """Dominant periodogram peak + ACF at that lag."""
    n = len(y)
    d = y - y.mean()
    if np.std(d) <= 0:
        return None, 0.0, 0.0
    # remove linear trend so a ramp is not read as a very long cycle
    d = signal.detrend(d)
    freqs, power = signal.periodogram(d)
    if len(power) < 4:
        return None, 0.0, 0.0
    power[0] = 0.0
    valid = (freqs > 1.0 / (n / 2.0)) & (freqs <= 0.5)
    if not valid.any():
        return None, 0.0, 0.0
    pv, fv = power[valid], freqs[valid]
    i = int(np.argmax(pv))
    ratio = float(pv[i] / (pv.sum() + 1e-12))
    period = float(1.0 / fv[i])
    lag = int(round(period))
    if lag < 2 or lag >= n // 2:
        return None, ratio, 0.0
    a = d[:-lag]
    b = d[lag:]
    acf = float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 and np.std(a) > 0 and np.std(b) > 0 else 0.0
    return period, ratio, acf


def _changepoint(y):
    """Largest single level shift and its magnitude in residual-SD units."""
    n = len(y)
    try:
        import ruptures as rpt
        algo = rpt.Binseg(model="l2", min_size=max(5, n // 10)).fit(y)
        bkps = algo.predict(n_bkps=1)
        cp = int(bkps[0])
    except Exception:
        best, cp = -1, n // 2
        for c in range(max(5, n // 10), n - max(5, n // 10)):
            d = abs(y[:c].mean() - y[c:].mean())
            if d > best:
                best, cp = d, c
    if cp <= 3 or cp >= n - 3:
        return None, 0.0, 0.0, 0.0, None, 0.0
    a, b = y[:cp], y[cp:]
    delta = float(b.mean() - a.mean())
    within = float(np.sqrt((a.var(ddof=1) * len(a) + b.var(ddof=1) * len(b)) / n)) if n > 2 else 0.0
    snr = float(abs(delta) / (within + 1e-12))

    # R^2 of a two-level step model, for head-to-head comparison with a linear fit.
    # A pure ramp also yields a high-SNR "changepoint", so SNR alone cannot separate
    # a regime shift from a trend -- the model comparison below is what does.
    fitted = np.r_[np.full(len(a), a.mean()), np.full(len(b), b.mean())]
    sst = float(((y - y.mean()) ** 2).sum())
    step_r2 = 1 - float(((y - fitted) ** 2).sum()) / (sst + 1e-12)

    # residual trend after removing the step: ~0 for a true shift, still sloped for a ramp
    destep = y - fitted

    # Abruptness: how much of the total level change happens in a narrow window
    # around the breakpoint. A true step concentrates it there (~1.0); a smooth
    # rise-then-fall spreads it out (<<1) yet can still defeat the destep test,
    # because that test only looks for a *monotonic* residual trend.
    w = max(2, n // 40)
    if cp - w >= 0 and cp + w <= n and abs(delta) > 1e-12:
        local = abs(float(y[cp:cp + w].mean() - y[cp - w:cp].mean()))
        abruptness = float(local / (abs(delta) + 1e-12))
    else:
        abruptness = 0.0
    return cp, delta, snr, step_r2, destep, abruptness


def verify_timeseries(t, y):
    t = np.asarray(t, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < MIN_N_TS:
        return None, {}, "too_few_points"
    order = np.argsort(t)
    t, y = t[order], y[order]
    if len(np.unique(t)) != len(t):
        return None, {}, "duplicate_timestamps"
    if len(t) > MAX_N_TS:
        idx = np.linspace(0, len(t) - 1, MAX_N_TS).astype(int)
        t, y = t[idx], y[idx]
    n = len(y)
    if np.std(y) <= 0:
        return None, {}, "zero_variance"

    tau, mk_p, mk_slope, mk_trend = _mk(y)
    period, seas_ratio, seas_acf = _seasonality(y)
    cp_idx, cp_delta, cp_snr, step_r2, destep, abruptness = _changepoint(y)
    tau_destep = _mk(destep)[0] if destep is not None and len(destep) >= MIN_N_TS else None

    # robust point anomalies on the detrended series
    det = signal.detrend(y)
    med = float(np.median(det))
    mad = float(np.median(np.abs(det - med)))
    rz = 0.6745 * (det - med) / (mad + 1e-12)
    anom = np.where(np.abs(rz) > 6.0)[0]
    max_abs_rz = float(np.max(np.abs(rz)))

    # trend after removing anomalies (separates a true spike from a trend)
    keep = np.setdiff1d(np.arange(n), anom)
    tau_clean = tau
    if len(anom) and len(keep) >= MIN_N_TS:
        tau_clean, _, _, _ = _mk(y[keep])

    try:
        from statsmodels.tsa.stattools import adfuller
        adf_p = _f(adfuller(y, autolag="AIC")[1])
    except Exception:
        adf_p = None

    lin_slope, lin_int = np.polyfit(np.arange(n), y, 1)
    lin_r2 = 1 - float(((y - (lin_slope * np.arange(n) + lin_int)) ** 2).sum()) / \
        (float(((y - y.mean()) ** 2).sum()) + 1e-12)

    s = dict(
        n=n, t_start=_f(t[0]), t_end=_f(t[-1]),
        mean=_f(y.mean()), std=_f(y.std(ddof=1)),
        min=_f(y.min()), max=_f(y.max()),
        first_value=_f(y[0]), last_value=_f(y[-1]),
        pct_change_total=_f(100.0 * (y[-1] - y[0]) / (abs(y[0]) + 1e-12)),
        mk_tau=tau, mk_p=mk_p, mk_sen_slope=mk_slope, mk_trend=mk_trend,
        mk_tau_excl_anomalies=tau_clean,
        ols_slope=_f(lin_slope), ols_r2=_f(lin_r2), adf_p=adf_p,
        seasonal_period=_f(period), seasonal_power_ratio=_f(seas_ratio), seasonal_acf=_f(seas_acf),
        changepoint_index=cp_idx, changepoint_delta=_f(cp_delta), changepoint_snr=_f(cp_snr),
        step_model_r2=_f(step_r2), tau_after_step_removal=tau_destep,
        changepoint_abruptness=_f(abruptness),
        anomaly_indices=[int(i) for i in anom[:12]], anomaly_count=int(len(anom)),
        max_abs_robust_z=_f(max_abs_rz),
    )

    tau = tau or 0.0
    mk_p = mk_p if mk_p is not None else 1.0
    tau_clean = tau_clean or 0.0
    td = abs(tau_destep) if tau_destep is not None else 1.0
    seas_ok = (seas_ratio > 0.30 and seas_acf > 0.35 and period is not None)
    anom_cap = max(3, int(0.02 * n))
    # a monotone ramp is better explained by a line, a regime shift by two levels
    step_wins = (step_r2 - lin_r2) > 0.10
    line_wins = (lin_r2 - step_r2) > -0.02

    m = []
    if mk_p < 0.01 and tau > 0.35 and not seas_ok and len(anom) == 0 and line_wins:
        m.append("ts_increasing")
    if mk_p < 0.01 and tau < -0.35 and not seas_ok and len(anom) == 0 and line_wins:
        m.append("ts_decreasing")
    if seas_ok and abs(tau) < 0.25:
        m.append("ts_seasonal")
    if (cp_idx is not None and cp_snr > 2.5 and step_wins and td < 0.20
            and abruptness > 0.55 and not seas_ok and len(anom) == 0):
        m.append("ts_changepoint")
    if 1 <= len(anom) <= anom_cap and max_abs_rz > 6.0 and abs(tau_clean) < 0.25 and not seas_ok:
        m.append("ts_anomaly_spike")

    label, why = _resolve(m)
    return label, s, why


# ==========================================================================
# FAMILY D — group comparison / composition
# ==========================================================================
MIN_N_GRP, MIN_PER_GRP = 60, 10
MAX_GROUPS = 8


def verify_grouped(values, groups):
    values = np.asarray(values, float).ravel()
    groups = np.asarray(groups).ravel()
    ok = np.isfinite(values)
    values, groups = values[ok], groups[ok]
    if len(values) < MIN_N_GRP:
        return None, {}, "too_few_points"

    labs, counts = np.unique(groups, return_counts=True)
    keep = labs[counts >= MIN_PER_GRP]
    if not (2 <= len(keep) <= MAX_GROUPS):
        return None, {}, "bad_group_count"
    sel = np.isin(groups, keep)
    values, groups = values[sel], groups[sel]
    n, k = len(values), len(keep)
    if n < MIN_N_GRP or np.std(values) <= 0:
        return None, {}, "degenerate"

    samples = [values[groups == g] for g in keep]
    try:
        H, kw_p = stats.kruskal(*samples)
    except Exception:
        return None, {}, "kruskal_failed"
    try:
        F, an_p = stats.f_oneway(*samples)
    except Exception:
        F = an_p = None

    # epsilon-squared effect size for Kruskal-Wallis
    eps2 = float((H - k + 1) / (n - k)) if n > k else 0.0
    eps2 = max(0.0, min(1.0, eps2))

    grand = values.mean()
    ssb = float(sum(len(s_) * (s_.mean() - grand) ** 2 for s_ in samples))
    sst = float(((values - grand) ** 2).sum())
    eta2 = ssb / (sst + 1e-12)

    try:
        lev_stat, lev_p = stats.levene(*samples)
    except Exception:
        lev_stat = lev_p = None

    means = {str(g): _f(s_.mean()) for g, s_ in zip(keep, samples)}
    medians = {str(g): _f(np.median(s_)) for g, s_ in zip(keep, samples)}
    gm = [(str(g), float(np.median(s_))) for g, s_ in zip(keep, samples)]
    gm.sort(key=lambda p: p[1])

    d = None
    if k == 2:
        a, b = samples
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (n - 2))
        d = _f((b.mean() - a.mean()) / (sp + 1e-12))

    s = dict(
        n=n, n_groups=k, group_sizes={str(g): int((groups == g).sum()) for g in keep},
        group_means=means, group_medians=medians,
        kruskal_H=_f(H), kruskal_p=_f(kw_p),
        anova_F=_f(F), anova_p=_f(an_p),
        epsilon_squared=_f(eps2), eta_squared=_f(eta2),
        levene_stat=_f(lev_stat), levene_p=_f(lev_p), cohens_d=d,
        lowest_group=gm[0][0], highest_group=gm[-1][0],
        median_gap=_f(gm[-1][1] - gm[0][1]),
        overall_mean=_f(grand), overall_median=_f(np.median(values)),
    )

    m = []
    if kw_p < 0.001 and eps2 > 0.14:
        m.append("grp_significant_diff")
    # Cohen: epsilon-squared below 0.01 is negligible, 0.01-0.06 small.
    # Paired with p > 0.30 this is a defensible "no difference" claim.
    if kw_p > 0.30 and eps2 < 0.02 and eta2 < 0.03:
        m.append("grp_no_diff")

    label, why = _resolve(m)
    return label, s, why


def verify_composition(labels):
    """Categorical composition -> grp_dominant_category."""
    labels = np.asarray([str(v) for v in labels if v is not None and str(v) != "nan"])
    n = len(labels)
    if n < MIN_N_GRP:
        return None, {}, "too_few_points"
    cats, counts = np.unique(labels, return_counts=True)
    k = len(cats)
    if not (3 <= k <= 12):
        return None, {}, "bad_category_count"

    order = np.argsort(-counts)
    cats, counts = cats[order], counts[order]
    p = counts / counts.sum()
    top_share = float(p[0])
    ent = float(-(p * np.log(p + 1e-12)).sum())
    norm_ent = float(ent / np.log(k))
    gini = float(1 - (p ** 2).sum())

    s = dict(
        n=n, n_categories=k,
        counts={str(c): int(v) for c, v in zip(cats, counts)},
        shares={str(c): _f(v) for c, v in zip(cats, p)},
        top_category=str(cats[0]), top_share=_f(top_share),
        second_category=str(cats[1]), second_share=_f(p[1]),
        shannon_entropy=_f(ent), normalised_entropy=_f(norm_ent), gini_impurity=_f(gini),
    )

    m = []
    if top_share > 0.55 and norm_ent < 0.75:
        m.append("grp_dominant_category")

    label, why = _resolve(m)
    return label, s, why


def _selftest_cases(rng):
    """Generators whose intended class is known by construction."""
    n = 400
    x = rng.normal(0, 1, 300)
    xu = rng.uniform(0.2, 4, 300)
    tt = np.arange(120)
    spike = rng.normal(0, 1, 120)
    spike[rng.integers(30, 90)] = 22.0
    g = np.repeat(list("ABC"), 60)
    cases = [
        ("dist_normal", lambda: verify_univariate(rng.normal(0, 1, n))),
        ("dist_right_skew", lambda: verify_univariate(rng.lognormal(0, 0.9, n))),
        ("dist_left_skew", lambda: verify_univariate(-rng.lognormal(0, 0.9, n))),
        ("dist_bimodal", lambda: verify_univariate(np.r_[rng.normal(-3, .7, 200), rng.normal(3, .7, 200)])),
        # Student-t df=4: symmetric (skew 0) with strong excess kurtosis. df<=3
        # has undefined skew and leaks into the skew classes; df>=5 is too mild.
        ("dist_heavy_tail", lambda: verify_univariate(rng.standard_t(4, 600))),
        ("dist_uniform", lambda: verify_univariate(rng.uniform(0, 1, 500))),
        ("rel_strong_positive", lambda: verify_bivariate(x, 2 * x + rng.normal(0, .6, 300))),
        ("rel_strong_negative", lambda: verify_bivariate(x, -2 * x + rng.normal(0, .6, 300))),
        ("rel_none", lambda: verify_bivariate(x, rng.normal(0, 1, 300))),
        ("rel_nonlinear_monotonic", lambda: verify_bivariate(xu, np.exp(3 * xu) + rng.normal(0, 400, 300))),
        ("rel_nonlinear_nonmonotonic", lambda: verify_bivariate(x, x ** 2 + rng.normal(0, .5, 300))),
        ("rel_heteroscedastic", lambda: verify_bivariate(xu, 2 * xu + rng.normal(0, 1, 300) * xu ** 2)),
        ("ts_increasing", lambda: verify_timeseries(tt, 0.4 * tt + rng.normal(0, 3, 120))),
        ("ts_decreasing", lambda: verify_timeseries(tt, -0.4 * tt + rng.normal(0, 3, 120))),
        ("ts_seasonal", lambda: verify_timeseries(tt, 10 * np.sin(2 * np.pi * tt / 12) + rng.normal(0, 1.2, 120))),
        ("ts_changepoint", lambda: verify_timeseries(tt, np.r_[rng.normal(0, 1, 60), rng.normal(7, 1, 60)])),
        ("ts_anomaly_spike", lambda: verify_timeseries(tt, spike)),
        ("grp_significant_diff", lambda: verify_grouped(
            np.r_[rng.normal(0, 1, 60), rng.normal(3, 1, 60), rng.normal(6, 1, 60)], g)),
        ("grp_no_diff", lambda: verify_grouped(rng.normal(0, 1, 180), g)),
        ("grp_dominant_category", lambda: verify_composition(np.r_[["A"] * 140, ["B"] * 40, ["C"] * 20])),
    ]
    return cases


def run_selftest(trials=10):
    """Generator->class fidelity. Reports hit rate per class over many draws."""
    from collections import Counter
    hits = Counter()
    misses = {}
    for seed in range(trials):
        rng = np.random.default_rng(seed)
        for exp, fn in _selftest_cases(rng):
            got, _s, why = fn()
            if got == exp:
                hits[exp] += 1
            else:
                misses.setdefault(exp, Counter())[str(got or why)] += 1

    names = [c for c, _ in _selftest_cases(np.random.default_rng(0))]
    print(f"\n{'class':32s} {'hit rate':>10s}   most common miss")
    print("-" * 78)
    total = 0
    for c in names:
        h = hits[c]
        total += h
        miss = ""
        if c in misses:
            k, v = misses[c].most_common(1)[0]
            miss = f"{k} ({v})"
        flag = "" if h == trials else ("  <-- " if h < trials * 0.8 else "  ~ ")
        print(f"{c:32s} {h:>4d}/{trials:<5d}   {miss}{flag}")
    print("-" * 78)
    print(f"overall generator->class fidelity: {total}/{len(names) * trials} "
          f"({100.0 * total / (len(names) * trials):.1f}%)")
    print(f"registry: {len(CLASSES)} classes across {len(FAMILIES)} families")
    return total / (len(names) * trials)


if __name__ == "__main__":
    _rate = run_selftest(trials=10)
    # marker so watch_progress.py can see this stage completed
    import pathlib
    _c = pathlib.Path(__file__).resolve().parent.parent / "cache"
    _c.mkdir(exist_ok=True)
    (_c / "selftest.marker").write_text(
        __import__("json").dumps({"fidelity": _rate, "classes": len(CLASSES)}))


def _legacy_single_shot():
    rng = np.random.default_rng(0)
    checks = [
        ("dist_normal", verify_univariate(rng.normal(0, 1, 400))),
        ("dist_right_skew", verify_univariate(rng.lognormal(0, 0.9, 400))),
        ("dist_left_skew", verify_univariate(-rng.lognormal(0, 0.9, 400))),
        ("dist_bimodal", verify_univariate(np.r_[rng.normal(-3, 0.7, 200), rng.normal(3, 0.7, 200)])),
        ("dist_heavy_tail", verify_univariate(rng.standard_t(2.5, 500))),
        ("dist_uniform", verify_univariate(rng.uniform(0, 1, 500))),
    ]
    x = rng.normal(0, 1, 300)
    checks += [
        ("rel_strong_positive", verify_bivariate(x, 2 * x + rng.normal(0, 0.6, 300))),
        ("rel_strong_negative", verify_bivariate(x, -2 * x + rng.normal(0, 0.6, 300))),
        ("rel_none", verify_bivariate(x, rng.normal(0, 1, 300))),
    ]
    xu = rng.uniform(0.2, 4, 300)
    checks += [
        ("rel_nonlinear_monotonic", verify_bivariate(xu, np.exp(1.6 * xu) + rng.normal(0, 3, 300))),
        ("rel_nonlinear_nonmonotonic", verify_bivariate(x, x ** 2 + rng.normal(0, 0.5, 300))),
        ("rel_heteroscedastic", verify_bivariate(xu, 2 * xu + rng.normal(0, 1, 300) * xu ** 2)),
    ]
    tt = np.arange(120)
    checks += [
        ("ts_increasing", verify_timeseries(tt, 0.4 * tt + rng.normal(0, 3, 120))),
        ("ts_decreasing", verify_timeseries(tt, -0.4 * tt + rng.normal(0, 3, 120))),
        ("ts_seasonal", verify_timeseries(tt, 10 * np.sin(2 * np.pi * tt / 12) + rng.normal(0, 1.2, 120))),
        ("ts_changepoint", verify_timeseries(tt, np.r_[rng.normal(0, 1, 60), rng.normal(7, 1, 60)])),
    ]
    spike = rng.normal(0, 1, 120).copy()
    spike[70] = 22
    checks += [("ts_anomaly_spike", verify_timeseries(tt, spike))]
    g = np.repeat(list("ABC"), 60)
    checks += [
        ("grp_significant_diff", verify_grouped(np.r_[rng.normal(0, 1, 60), rng.normal(3, 1, 60), rng.normal(6, 1, 60)], g)),
        ("grp_no_diff", verify_grouped(rng.normal(0, 1, 180), g)),
        ("grp_dominant_category", verify_composition(np.r_[["A"] * 140, ["B"] * 40, ["C"] * 20])),
    ]

    print(f"{'expected':32s} {'got':32s} ok")
    print("-" * 72)
    npass = 0
    for exp, (got, _st, why) in checks:
        good = got == exp
        npass += good
        print(f"{exp:32s} {str(got or why):32s} {'PASS' if good else 'FAIL'}")
    print("-" * 72)
    print(f"{npass}/{len(checks)} generators land on their intended class")
    print(f"registry: {len(CLASSES)} classes across {len(FAMILIES)} families")
