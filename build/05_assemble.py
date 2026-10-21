"""
05_assemble.py — Balance, split, and build the ground-truth records.

Selection policy
  * Exactly TARGET samples per class (perfect balance).
  * Provenance is balanced *within* each class (up to 500 real + 500 synthetic)
    rather than real-first. Real-first would leave some classes ~100% real and
    others ~100% synthetic, letting a model key on provenance artefacts (column
    naming, value scales) instead of the statistical pattern.
  * Splits are source-disjoint for real data: a source table is assigned wholly
    to train, val, or test, so no chart derived from `openml/61_iris` can appear
    in both train and test.

Every record carries the full statistical ground truth: the test that licensed
the label, the complete statistic dictionary, the underlying data table, and a
step-by-step reasoning chain in which every claim cites a computed number.

Output: cache/records.jsonl
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tax  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
OUT = CACHE / "records.jsonl"

TARGET = 1000
REAL_CAP_PER_CLASS = 500
SPLIT_FRACS = {"train": 0.70, "val": 0.10, "test": 0.20}
SEED = 20260815


def _h(s):
    return int(hashlib.blake2b(str(s).encode(), digest_size=8).hexdigest(), 16)


def data_fingerprint(c):
    """Content hash of the plotted values, for exact-duplicate removal."""
    parts = []
    for k in sorted(c["data"]):
        v = c["data"][k]
        if v and isinstance(v[0], (int, float)):
            parts.append(k + ":" + ",".join(f"{float(z):.6g}" for z in v))
        else:
            parts.append(k + ":" + ",".join(map(str, v)))
    return hashlib.blake2b("|".join(parts).encode(), digest_size=16).hexdigest()


def assign_split(source_id):
    """Deterministic source -> split, so all views of a table land together."""
    u = (_h(("split", source_id)) % 10_000) / 10_000.0
    if u < SPLIT_FRACS["train"]:
        return "train"
    if u < SPLIT_FRACS["train"] + SPLIT_FRACS["val"]:
        return "val"
    return "test"


# --------------------------------------------------------------------------
# difficulty: how far the evidence sits from the decision threshold
# --------------------------------------------------------------------------
def evidence_margin(cls, st):
    """How far this sample's evidence sits beyond the decision threshold.

    Higher = more obvious. Returned as a raw statistic; the easy/medium/hard cut
    is made per class from the *observed* distribution (see assign_difficulty),
    not from fixed constants. Fixed constants do not work here: the verification
    gate already refuses weak cases -- a bimodal too subtle to trip Hartigan's
    dip test is not admitted at all -- so several classes have no attainable
    "hard" region on an absolute scale, and every sample piles up as easy.
    """
    g = st.get
    m = {
        "dist_normal": lambda: g("shapiro_p") or 0,
        "dist_right_skew": lambda: abs(g("skewness") or 0),
        "dist_left_skew": lambda: abs(g("skewness") or 0),
        "dist_bimodal": lambda: g("gmm_mode_separation") or 0,
        "dist_heavy_tail": lambda: g("excess_kurtosis") or 0,
        "dist_uniform": lambda: g("ks_uniform_p") or 0,
        "rel_strong_positive": lambda: abs(g("pearson_r") or 0),
        "rel_strong_negative": lambda: abs(g("pearson_r") or 0),
        "rel_none": lambda: -abs(g("pearson_r") or 0),
        "rel_nonlinear_monotonic": lambda: abs(g("spearman_rho") or 0) - abs(g("pearson_r") or 0),
        "rel_nonlinear_nonmonotonic": lambda: g("quadratic_r2") or 0,
        "rel_heteroscedastic": lambda: g("resid_spread_ratio") or 0,
        "ts_increasing": lambda: abs(g("mk_tau") or 0),
        "ts_decreasing": lambda: abs(g("mk_tau") or 0),
        "ts_seasonal": lambda: g("seasonal_acf") or 0,
        "ts_changepoint": lambda: g("changepoint_snr") or 0,
        "ts_anomaly_spike": lambda: g("max_abs_robust_z") or 0,
        "grp_significant_diff": lambda: g("epsilon_squared") or 0,
        "grp_no_diff": lambda: g("kruskal_p") or 0,
        "grp_dominant_category": lambda: g("top_share") or 0,
    }.get(cls)
    try:
        return float(m()) if m else 0.0
    except (TypeError, ValueError):
        return 0.0


def assign_difficulty(chosen, cls):
    """Tertile-split the class's own evidence margins into easy/medium/hard."""
    vals = np.array([evidence_margin(cls, c["stats"]) for c in chosen], dtype=float)
    lo, hi = np.quantile(vals, [1 / 3, 2 / 3])
    out = []
    for v in vals:
        out.append("hard" if v <= lo else ("medium" if v <= hi else "easy"))
    return out


# --------------------------------------------------------------------------
# evidence-grounded reasoning chain — every claim cites a computed number
# --------------------------------------------------------------------------
def _n(v, p=2):
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 1e6 or (a < 1e-3 and a > 0):
        return f"{v:.{p}e}"
    return f"{v:,.{p}f}"


def _pv(p):
    if p is None:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def reasoning_chain(cls, st, cols, axes):
    """(steps, insight) — steps are observation -> statistic -> test -> conclusion."""
    g = st.get
    xa = axes.get("x", cols[0] if cols else "x")
    ya = axes.get("y", cols[-1] if cols else "y")
    S = []

    if cls.startswith("dist_"):
        v = xa
        S.append(f"The chart shows the distribution of {v} over n = {g('n')} observations.")
        S.append(f"Centre: mean = {_n(g('mean'))}, median = {_n(g('median'))}; "
                 f"spread: SD = {_n(g('std'))}, IQR = {_n(g('iqr'))} "
                 f"(Q1 = {_n(g('q1'))}, Q3 = {_n(g('q3'))}).")
        S.append(f"Shape: skewness = {_n(g('skewness'))} (robust Bowley "
                 f"{_n(g('bowley_skew'))}), excess kurtosis = {_n(g('excess_kurtosis'))}.")
        S.append(f"Normality: Shapiro-Wilk W = {_n(g('shapiro_stat'), 3)}, "
                 f"p {_pv(g('shapiro_p'))}. Hartigan dip p {_pv(g('dip_p'))}.")
        if cls == "dist_normal":
            ins = (f"{v} is approximately normally distributed. Shapiro-Wilk does not reject "
                   f"normality (p {_pv(g('shapiro_p'))}), skewness ({_n(g('skewness'))}) and excess "
                   f"kurtosis ({_n(g('excess_kurtosis'))}) are both near zero, and only "
                   f"{g('outlier_count')} of {g('n')} points ({_n(g('outlier_pct'),1)}%) fall outside "
                   f"the 1.5xIQR fences. Mean ({_n(g('mean'))}) and median ({_n(g('median'))}) "
                   f"nearly coincide, so parametric summaries are appropriate.")
        elif cls in ("dist_right_skew", "dist_left_skew"):
            side = "right" if cls == "dist_right_skew" else "left"
            rel = "above" if side == "right" else "below"
            ins = (f"{v} is {side}-skewed (skewness = {_n(g('skewness'))}, Bowley "
                   f"{_n(g('bowley_skew'))}), with a long {side} tail. The mean "
                   f"({_n(g('mean'))}) sits {rel} the median ({_n(g('median'))}), so the mean "
                   f"overstates the typical value; report the median instead. Shapiro-Wilk "
                   f"rejects normality (p {_pv(g('shapiro_p'))}). A log or Box-Cox transform "
                   f"would be appropriate before any parametric test.")
        elif cls == "dist_bimodal":
            ins = (f"{v} is bimodal: Hartigan's dip test rejects unimodality "
                   f"(p {_pv(g('dip_p'))}) and a two-component Gaussian mixture improves BIC by "
                   f"{_n((g('gmm_bic_1') or 0) - (g('gmm_bic_2') or 0), 1)} with modes separated by "
                   f"{_n(g('gmm_mode_separation'))} pooled SD. A single mean ({_n(g('mean'))}) is "
                   f"misleading because it falls between the two modes; the data most likely mixes "
                   f"two sub-populations that should be modelled separately.")
        elif cls == "dist_heavy_tail":
            ins = (f"{v} is heavy-tailed: excess kurtosis = {_n(g('excess_kurtosis'))} with "
                   f"{g('outlier_count')} points ({_n(g('outlier_pct'),1)}%) beyond the 1.5xIQR "
                   f"fences [{_n(g('iqr_lower_fence'))}, {_n(g('iqr_upper_fence'))}], while robust "
                   f"skew stays near zero ({_n(g('bowley_skew'))}) so the tails are symmetric. "
                   f"The SD ({_n(g('std'))}) is inflated by extremes -- prefer the IQR and median, "
                   f"and use rank-based tests.")
        else:
            ins = (f"{v} is approximately uniform across "
                   f"[{_n(g('min'))}, {_n(g('max'))}]: a KS test against the uniform does not "
                   f"reject (p {_pv(g('ks_uniform_p'))}) and excess kurtosis is "
                   f"{_n(g('excess_kurtosis'))}, the flat/platykurtic signature. No value range is "
                   f"preferred, so there is no meaningful mode and central tendency carries little "
                   f"information here.")

    elif cls.startswith("rel_"):
        S.append(f"The chart relates {xa} (x) to {ya} (y) over n = {g('n')} paired observations.")
        S.append(f"Linear association: Pearson r = {_n(g('pearson_r'),3)} (p {_pv(g('pearson_p'))}), "
                 f"linear R^2 = {_n(g('linear_r2'),3)}.")
        S.append(f"Rank association: Spearman rho = {_n(g('spearman_rho'),3)}, "
                 f"Kendall tau = {_n(g('kendall_tau'),3)}.")
        S.append(f"Curvature: quadratic R^2 = {_n(g('quadratic_r2'),3)} (t on the squared term = "
                 f"{_n(g('curvature_t'),1)}); Breusch-Pagan p {_pv(g('bp_p'))}.")
        if cls in ("rel_strong_positive", "rel_strong_negative"):
            d = "positive" if cls == "rel_strong_positive" else "negative"
            dr = "increases" if d == "positive" else "decreases"
            ins = (f"{ya} shows a strong {d} linear relationship with {xa} "
                   f"(r = {_n(g('pearson_r'),3)}, p {_pv(g('pearson_p'))}), explaining "
                   f"{_n(100*(g('linear_r2') or 0),1)}% of the variance. The fitted slope is "
                   f"{_n(g('linear_slope'),4)}: each one-unit rise in {xa} {dr} {ya} by about "
                   f"{_n(abs(g('linear_slope') or 0),4)}. Spearman ({_n(g('spearman_rho'),3)}) "
                   f"agrees closely with Pearson, confirming the association is genuinely linear "
                   f"rather than driven by curvature or outliers.")
        elif cls == "rel_none":
            ins = (f"There is no detectable relationship between {xa} and {ya}. Pearson "
                   f"(r = {_n(g('pearson_r'),3)}, p {_pv(g('pearson_p'))}) and Spearman "
                   f"(rho = {_n(g('spearman_rho'),3)}) are both near zero, and a quadratic fit adds "
                   f"nothing (R^2 = {_n(g('quadratic_r2'),3)}), so a curved pattern is ruled out "
                   f"too. The scatter is unstructured; {xa} carries no predictive information "
                   f"about {ya} in this sample.")
        elif cls == "rel_nonlinear_monotonic":
            ins = (f"{ya} rises monotonically with {xa} but along a curve, not a line. Spearman "
                   f"({_n(g('spearman_rho'),3)}) far exceeds Pearson ({_n(g('pearson_r'),3)}) -- a "
                   f"gap of {_n(abs((g('spearman_rho') or 0))-abs((g('pearson_r') or 0)),3)} -- and "
                   f"the linear fit captures only {_n(100*(g('linear_r2') or 0),1)}% of the "
                   f"variance. Pearson understates the true dependence; the ordering is almost "
                   f"perfectly preserved, so a monotone or log transform would linearise it.")
        elif cls == "rel_nonlinear_nonmonotonic":
            shape = "inverted-U" if (g('curvature_coef') or 0) < 0 else "U"
            ext = "maximum" if shape == "inverted-U" else "minimum"
            ins = (f"{ya} has a {shape}-shaped dependence on {xa}: a quadratic fit explains "
                   f"{_n(100*(g('quadratic_r2') or 0),1)}% of the variance with a highly significant "
                   f"squared term (t = {_n(g('curvature_t'),1)}), while linear correlation is "
                   f"near zero (r = {_n(g('pearson_r'),3)}). Reporting Pearson alone would wrongly "
                   f"conclude 'no relationship'. There is an interior {ext} -- the effect of {xa} "
                   f"reverses sign across the range.")
        else:
            ins = (f"The relationship between {xa} and {ya} is heteroscedastic: residual variance "
                   f"grows with {xa}, giving a fan shape. Breusch-Pagan rejects constant variance "
                   f"(LM = {_n(g('bp_lm_stat'),1)}, p {_pv(g('bp_p'))}) and residual SD in the top "
                   f"tercile of {xa} is {_n(g('resid_spread_ratio'),1)}x that in the bottom "
                   f"tercile. Ordinary least squares remains unbiased but its standard errors are "
                   f"wrong -- use robust (Huber-White) errors or a variance-stabilising transform.")

    elif cls.startswith("ts_"):
        S.append(f"The chart is a time series of {ya} against {xa}, n = {g('n')} points "
                 f"spanning {_n(g('t_start'),0)} to {_n(g('t_end'),0)}.")
        S.append(f"Level: first = {_n(g('first_value'))}, last = {_n(g('last_value'))} "
                 f"({_n(g('pct_change_total'),1)}% total change); mean = {_n(g('mean'))}, "
                 f"SD = {_n(g('std'))}.")
        S.append(f"Trend: Mann-Kendall tau = {_n(g('mk_tau'),3)} (p {_pv(g('mk_p'))}), "
                 f"Sen slope = {_n(g('mk_sen_slope'),4)} per step, OLS R^2 = {_n(g('ols_r2'),3)}.")
        S.append(f"Structure: seasonal power ratio = {_n(g('seasonal_power_ratio'),3)}, "
                 f"changepoint SNR = {_n(g('changepoint_snr'),2)}, "
                 f"max robust |z| = {_n(g('max_abs_robust_z'),1)}.")
        if cls in ("ts_increasing", "ts_decreasing"):
            d = "upward" if cls == "ts_increasing" else "downward"
            ins = (f"{ya} shows a sustained {d} trend across the observed window. The "
                   f"non-parametric Mann-Kendall test confirms a monotonic trend "
                   f"(tau = {_n(g('mk_tau'),3)}, p {_pv(g('mk_p'))}), with a Sen slope of "
                   f"{_n(g('mk_sen_slope'),4)} per step -- a total change of "
                   f"{_n(g('pct_change_total'),1)}% from {_n(g('first_value'))} to "
                   f"{_n(g('last_value'))}. A two-level step model does not explain the series "
                   f"better than a line, so this is a gradual trend rather than a regime shift.")
        elif cls == "ts_seasonal":
            ins = (f"{ya} is dominated by a repeating cycle of roughly "
                   f"{_n(g('seasonal_period'),1)} steps. That frequency carries "
                   f"{_n(100*(g('seasonal_power_ratio') or 0),1)}% of the detrended spectral power "
                   f"and autocorrelation at the implied lag is {_n(g('seasonal_acf'),2)}. Net trend "
                   f"is negligible (Mann-Kendall tau = {_n(g('mk_tau'),3)}), so period-over-period "
                   f"comparisons must be made against the same phase of the cycle, not against the "
                   f"immediately preceding point.")
        elif cls == "ts_changepoint":
            ins = (f"{ya} undergoes an abrupt level shift at index {g('changepoint_index')} "
                   f"(around {xa} = {_n(g('t_start'),0)} + {g('changepoint_index')} steps). The mean "
                   f"moves by {_n(g('changepoint_delta'))}, which is {_n(g('changepoint_snr'),1)}x "
                   f"the within-segment SD, and a two-level step model fits better than a straight "
                   f"line (step R^2 = {_n(g('step_model_r2'),3)} vs linear "
                   f"{_n(g('ols_r2'),3)}). Once the step is removed the residual trend vanishes "
                   f"(tau = {_n(g('tau_after_step_removal'),3)}), so this is a discrete regime "
                   f"change -- fitting a single trend across the break would be misleading.")
        else:
            k = g('anomaly_count')
            ins = (f"{ya} is otherwise stable but contains {k} isolated "
                   f"anomal{'y' if k == 1 else 'ies'} at index "
                   f"{', '.join(map(str, g('anomaly_indices') or []))}. The largest deviates "
                   f"{_n(g('max_abs_robust_z'),1)} robust standard deviations (median/MAD) from the "
                   f"local level. Excluding those points there is no trend "
                   f"(tau = {_n(g('mk_tau_excl_anomalies'),3)}), confirming the spikes are "
                   f"point anomalies rather than the start of a shift; they should be investigated "
                   f"or winsorised before modelling.")

    elif cls == "grp_dominant_category":
        S.append(f"The chart shows the composition of {xa} over n = {g('n')} records in "
                 f"{g('n_categories')} categories.")
        S.append("Counts: " + ", ".join(f"{k} = {v}" for k, v in list((g('counts') or {}).items())[:6]) + ".")
        S.append(f"Concentration: top share = {_n(100*(g('top_share') or 0),1)}%, normalised "
                 f"Shannon entropy = {_n(g('normalised_entropy'),3)}, Gini impurity = "
                 f"{_n(g('gini_impurity'),3)}.")
        ins = (f"The composition of {xa} is dominated by '{g('top_category')}', which accounts for "
               f"{_n(100*(g('top_share') or 0),1)}% of all {g('n')} records -- "
               f"{_n((g('top_share') or 0)/max(g('second_share') or 1e-9,1e-9),1)}x the next largest "
               f"category ('{g('second_category')}', {_n(100*(g('second_share') or 0),1)}%). "
               f"Normalised entropy is only {_n(g('normalised_entropy'),3)}, confirming strong "
               f"imbalance. Any model trained on this column needs class-imbalance handling, and "
               f"accuracy would be a misleading metric.")

    else:  # grouped comparison
        S.append(f"The chart compares {ya} across {g('n_groups')} levels of {xa}, "
                 f"n = {g('n')} observations.")
        S.append("Group medians: " + ", ".join(f"{k} = {_n(v)}" for k, v in list((g('group_medians') or {}).items())[:6]) + ".")
        S.append(f"Kruskal-Wallis H = {_n(g('kruskal_H'),2)}, p {_pv(g('kruskal_p'))}; "
                 f"epsilon-squared = {_n(g('epsilon_squared'),3)}, eta-squared = "
                 f"{_n(g('eta_squared'),3)}.")
        S.append(f"Variance homogeneity: Levene p {_pv(g('levene_p'))}.")
        if cls == "grp_significant_diff":
            ins = (f"{ya} differs substantially across levels of {xa}. Kruskal-Wallis is highly "
                   f"significant (H = {_n(g('kruskal_H'),2)}, p {_pv(g('kruskal_p'))}) and the "
                   f"effect is large (epsilon-squared = {_n(g('epsilon_squared'),3)}, well above the "
                   f"0.14 large-effect threshold). '{g('highest_group')}' has the highest median and "
                   f"'{g('lowest_group')}' the lowest, a gap of {_n(g('median_gap'))} "
                   f"-- so the difference is not only statistically detectable but practically "
                   f"meaningful.")
        else:
            ins = (f"{ya} does not differ meaningfully across levels of {xa}. Kruskal-Wallis is "
                   f"far from significant (H = {_n(g('kruskal_H'),2)}, p {_pv(g('kruskal_p'))}) and "
                   f"the effect size is negligible (epsilon-squared = "
                   f"{_n(g('epsilon_squared'),3)}, below the 0.01 threshold). The visible gap "
                   f"between group medians ({_n(g('median_gap'))}) is within sampling noise given "
                   f"the within-group spread; reading a real difference off this chart would be an "
                   f"error.")

    S.append(f"Conclusion: {tax.CLASSES[cls]['desc']} "
             f"(verified by {tax.CLASSES[cls]['test']}).")
    return S, ins


def build_qa(cls, st, cols, axes, insight):
    """Short QA pairs derived from the same verified numbers."""
    g = st.get
    qa = [{"q": "What is the main statistical finding in this chart?", "a": insight}]
    if cls.startswith("dist_"):
        qa += [
            {"q": "What are the mean and median?",
             "a": f"Mean = {_n(g('mean'))}, median = {_n(g('median'))}."},
            {"q": "How many outliers are there by the 1.5xIQR rule?",
             "a": f"{g('outlier_count')} of {g('n')} points ({_n(g('outlier_pct'),1)}%), "
                  f"outside [{_n(g('iqr_lower_fence'))}, {_n(g('iqr_upper_fence'))}]."},
            {"q": "Is the distribution normal?",
             "a": f"Shapiro-Wilk p {_pv(g('shapiro_p'))}, so normality is "
                  f"{'not rejected' if (g('shapiro_p') or 0) > 0.05 else 'rejected'}."},
        ]
    elif cls.startswith("rel_"):
        qa += [
            {"q": "What is the correlation between the two variables?",
             "a": f"Pearson r = {_n(g('pearson_r'),3)} (p {_pv(g('pearson_p'))}); "
                  f"Spearman rho = {_n(g('spearman_rho'),3)}."},
            {"q": "Is a linear model appropriate here?",
             "a": (f"Linear R^2 = {_n(g('linear_r2'),3)} vs quadratic R^2 = "
                   f"{_n(g('quadratic_r2'),3)}; Breusch-Pagan p {_pv(g('bp_p'))}.")},
        ]
    elif cls.startswith("ts_"):
        qa += [
            {"q": "Is there a trend, and how strong?",
             "a": f"Mann-Kendall tau = {_n(g('mk_tau'),3)}, p {_pv(g('mk_p'))}, "
                  f"Sen slope = {_n(g('mk_sen_slope'),4)} per step."},
            {"q": "How much did the series change overall?",
             "a": f"From {_n(g('first_value'))} to {_n(g('last_value'))}, "
                  f"a change of {_n(g('pct_change_total'),1)}%."},
        ]
    elif cls == "grp_dominant_category":
        qa += [{"q": "Which category dominates and by how much?",
                "a": f"'{g('top_category')}' at {_n(100*(g('top_share') or 0),1)}% of "
                     f"{g('n')} records."}]
    else:
        qa += [
            {"q": "Do the groups differ significantly?",
             "a": f"Kruskal-Wallis H = {_n(g('kruskal_H'),2)}, p {_pv(g('kruskal_p'))}, "
                  f"epsilon-squared = {_n(g('epsilon_squared'),3)}."},
            {"q": "Which group has the highest and lowest median?",
             "a": f"Highest: '{g('highest_group')}'; lowest: '{g('lowest_group')}' "
                  f"(gap {_n(g('median_gap'))})."},
        ]
    return qa


def main():
    rng = np.random.default_rng(SEED)
    pools = defaultdict(lambda: {"real": [], "synthetic": []})
    seen = set()
    n_dup = 0

    for fn in ("candidates_real.jsonl", "candidates_synth.jsonl"):
        p = CACHE / fn
        if not p.exists():
            continue
        with open(p) as fh:
            for line in tqdm(fh, desc=f"loading {fn}", unit="cand", ncols=100):
                c = json.loads(line)
                fp = data_fingerprint(c)
                if fp in seen:
                    n_dup += 1
                    continue
                seen.add(fp)
                c["fingerprint"] = fp
                pools[c["label"]][c["provenance"]].append(c)

    print(f"\nloaded pools (deduplicated, {n_dup:,} exact duplicates dropped)")

    records = []
    stats_rows = []
    for cls in tax.CLASS_NAMES:
        real = pools[cls]["real"]
        synth = pools[cls]["synthetic"]
        rng.shuffle(real)
        rng.shuffle(synth)

        # Real data must respect source-disjoint buckets: every view derived from
        # one source table lands in a single split. Synthetic samples each have a
        # unique source_id, so they carry no leakage risk and are drawn from one
        # shared pool -- which is also what lets every split fill exactly.
        real_by_split = {s: [] for s in SPLIT_FRACS}
        for c in real:
            real_by_split[assign_split(c["source_id"])].append(c)
        synth_pool = list(synth)

        chosen = []
        for split, frac in SPLIT_FRACS.items():
            want = int(round(TARGET * frac))
            want_real = min(len(real_by_split[split]), int(round(REAL_CAP_PER_CLASS * frac)))
            take_r = real_by_split[split][:want_real]
            take_s = synth_pool[: want - len(take_r)]
            synth_pool = synth_pool[len(take_s):]
            # synthetic exhausted -> take more real from this split's own bucket
            short = want - len(take_r) - len(take_s)
            if short > 0:
                take_r += real_by_split[split][want_real:want_real + short]
            for c in take_r + take_s:
                c["split"] = split
            chosen += take_r + take_s

        n_real = sum(1 for c in chosen if c["provenance"] == "real")
        stats_rows.append((cls, len(chosen), n_real, len(chosen) - n_real,
                           len(real), len(synth)))

        charts = tax.CLASSES[cls]["charts"]
        diffs = assign_difficulty(chosen, cls)
        for i, c in enumerate(chosen):
            st = c["stats"]
            steps, insight = reasoning_chain(cls, st, c["columns"], c["axis_labels"])
            sid = f"{cls}__{i:04d}"
            records.append({
                "sample_id": sid,
                "image": f"images/{cls}/{sid}.png",
                "split": c["split"],
                "label": cls,
                "family": tax.CLASSES[cls]["family"],
                "arity": c["arity"],
                "chart_type": charts[i % len(charts)],
                "difficulty": diffs[i],
                "evidence_margin": round(evidence_margin(cls, st), 6),
                "provenance": c["provenance"],
                "domain": c.get("domain"),      # subject area; needed by the bias audit
                "source_id": c["source_id"],
                "source": c["source"],
                "source_url": c.get("source_url"),
                "license": c.get("license"),
                "table": c.get("table"),
                "columns": c["columns"],
                "axis_labels": c["axis_labels"],
                "fingerprint": c["fingerprint"],
                # ---------------- GROUND TRUTH ----------------
                "ground_truth": {
                    "class": cls,
                    "class_description": tax.CLASSES[cls]["desc"],
                    "verification_test": tax.CLASSES[cls]["test"],
                    "statistics": st,
                    "data_table": c["data"],
                    "n_points": st.get("n"),
                    "reasoning_chain": steps,
                    "insight": insight,
                },
                "qa": build_qa(cls, st, c["columns"], c["axis_labels"], insight),
            })

    rng.shuffle(records)
    with open(OUT, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print(f"\n{'class':30s} {'total':>6s} {'real':>6s} {'synth':>6s} {'(pool r/s)':>16s}")
    print("-" * 72)
    for cls, tot, nr, ns, pr, ps in stats_rows:
        print(f"{cls:30s} {tot:>6,} {nr:>6,} {ns:>6,} {f'{pr:,}/{ps:,}':>16s}")
    print("-" * 72)
    tr = sum(r[2] for r in stats_rows)
    ts_ = sum(r[3] for r in stats_rows)
    print(f"{'TOTAL':30s} {len(records):>6,} {tr:>6,} {ts_:>6,}")
    print(f"\nprovenance mix: {100*tr/len(records):.1f}% real / {100*ts_/len(records):.1f}% synthetic")
    sp = Counter(r["split"] for r in records)
    print("splits: " + ", ".join(f"{k}={v:,}" for k, v in sorted(sp.items())))
    print(f"records -> {OUT}")


if __name__ == "__main__":
    main()
