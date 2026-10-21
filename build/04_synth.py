"""
04_synth.py — Controlled synthesis to top up classes real data cannot balance.

Public corpora contain plenty of right-skewed columns and almost no clean
inverted-U relationships, so a 1000/class balance is unreachable from real data
alone. Each generator here produces data whose property holds *by construction*,
then passes through the identical taxonomy gate used for the real track: if the
statistical test does not confirm the intended class, the sample is discarded.
Nothing is labelled by assumption.

Variables are drawn from a domain vocabulary (health, economics, environment,
engineering, education, retail) with plausible names, units and magnitudes so the
rendered charts do not read as obviously synthetic.

Output: cache/candidates_synth.jsonl
"""
import json
import math
import os
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tax  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
OUT = CACHE / "candidates_synth.jsonl"

TARGET = 1000
BUFFER = 1.25          # over-generate so assembly can dedup and still fill quota
# Every class gets a synthetic floor even when real data alone could fill it.
# Without this, classes like rel_none and ts_increasing end up ~100% real while
# rel_nonlinear_nonmonotonic is ~99% synthetic, and provenance becomes a
# shortcut feature that correlates with the label.
SYNTH_FLOOR = 620
MAX_TRIES_PER_HIT = 60

# --------------------------------------------------------------------------
# domain vocabulary — (name, unit, plausible centre, plausible spread)
# --------------------------------------------------------------------------
VOCAB = {
    "health": [
        ("Systolic blood pressure", "mmHg", 128, 16), ("Resting heart rate", "bpm", 72, 11),
        ("Body mass index", "kg/m^2", 26, 4.5), ("Serum cholesterol", "mg/dL", 198, 38),
        ("Haemoglobin", "g/dL", 13.6, 1.6), ("Fasting glucose", "mg/dL", 96, 18),
        ("Length of stay", "days", 4.2, 2.8), ("Vitamin D level", "ng/mL", 28, 9),
        ("Birth weight", "g", 3280, 480), ("Recovery time", "days", 14, 6),
    ],
    "economics": [
        ("GDP per capita", "USD", 18500, 9000), ("Household income", "USD", 54000, 21000),
        ("Unemployment rate", "%", 6.4, 2.4), ("Inflation rate", "%", 3.1, 1.8),
        ("House price", "USD thousand", 340, 120), ("Hourly wage", "USD", 24.5, 8.2),
        ("Export value", "USD million", 820, 350), ("Tax revenue", "% of GDP", 19.2, 5.1),
        ("Monthly expenditure", "USD", 2450, 780), ("Loan amount", "USD thousand", 185, 62),
    ],
    "environment": [
        ("CO2 emissions per capita", "tonnes", 5.2, 2.4), ("Annual rainfall", "mm", 940, 280),
        ("Mean temperature", "degC", 15.4, 6.1), ("PM2.5 concentration", "ug/m^3", 32, 14),
        ("Forest cover", "% of land", 31, 12), ("River discharge", "m^3/s", 480, 190),
        ("Soil moisture", "%", 24, 7), ("Wind speed", "m/s", 6.2, 2.1),
        ("Solar irradiance", "W/m^2", 620, 165), ("Water pH", "pH", 7.2, 0.6),
    ],
    "engineering": [
        ("Tensile strength", "MPa", 415, 78), ("Surface roughness", "um", 1.8, 0.7),
        ("Vibration amplitude", "mm/s", 3.4, 1.4), ("Fuel efficiency", "km/L", 14.6, 3.8),
        ("Bearing temperature", "degC", 68, 12), ("Compressive strength", "MPa", 38, 9),
        ("Power consumption", "kW", 142, 46), ("Cycle time", "s", 58, 17),
        ("Defect rate", "ppm", 320, 140), ("Pressure drop", "kPa", 27, 9),
    ],
    "education": [
        ("Exam score", "points", 68, 14), ("Study hours per week", "hours", 12.5, 5.4),
        ("Attendance rate", "%", 87, 9), ("Reading proficiency", "score", 512, 78),
        ("Class size", "students", 27, 7), ("Graduation rate", "%", 78, 12),
        ("Tuition fee", "USD", 12400, 5200), ("Library visits per term", "visits", 18, 9),
    ],
    "retail": [
        ("Units sold", "units", 480, 190), ("Basket value", "USD", 62.4, 24.8),
        ("Delivery time", "hours", 38, 16), ("Customer rating", "stars", 4.1, 0.7),
        ("Return rate", "%", 8.2, 3.4), ("Daily footfall", "visitors", 1240, 420),
        ("Discount applied", "%", 17, 8), ("Shelf stock level", "units", 260, 95),
    ],
}
DOMAINS = list(VOCAB)

CAT_SETS = [
    ("Region", ["North", "South", "East", "West", "Central"]),
    ("Treatment arm", ["Placebo", "Low dose", "High dose"]),
    ("Product line", ["Standard", "Premium", "Economy", "Bundle"]),
    ("Education level", ["Primary", "Secondary", "Tertiary"]),
    ("Machine", ["Line A", "Line B", "Line C", "Line D"]),
    ("Season", ["Winter", "Spring", "Summer", "Autumn"]),
    ("Income bracket", ["Low", "Lower-middle", "Upper-middle", "High"]),
    ("Site", ["Site 1", "Site 2", "Site 3", "Site 4", "Site 5"]),
    ("Channel", ["Online", "In-store", "Wholesale", "Partner"]),
    ("Age group", ["18-29", "30-44", "45-59", "60+"]),
]

TIME_AXES = [
    ("Year", lambda n, r: np.arange(1970 + r.integers(0, 30), 1970 + r.integers(0, 30) + n)),
    ("Month index", lambda n, r: np.arange(1, n + 1)),
    ("Quarter", lambda n, r: np.arange(1, n + 1)),
    ("Week", lambda n, r: np.arange(1, n + 1)),
    ("Day", lambda n, r: np.arange(1, n + 1)),
]


def pick_var(r, domain=None):
    d = domain or DOMAINS[r.integers(len(DOMAINS))]
    name, unit, centre, spread = VOCAB[d][r.integers(len(VOCAB[d]))]
    return d, name, unit, float(centre), float(spread)


def _lab(name, unit):
    return f"{name} ({unit})" if unit else name


# --------------------------------------------------------------------------
# generators: each returns (data_dict, columns, axis_labels, meta) or None
# --------------------------------------------------------------------------
def g_univariate(r, kind):
    d, name, unit, c, s = pick_var(r)
    n = int(r.integers(120, 601))

    if kind == "dist_normal":
        x = r.normal(c, s, n)
    elif kind in ("dist_right_skew", "dist_left_skew"):
        # Lognormal skewness is (e^v+2)*sqrt(e^v-1) for v=sigma^2. sigma in
        # [0.33,0.75] spans skew ~1.05 to ~3.3, i.e. from just past the 1.0 gate
        # to clearly skewed. The old [0.55,1.05] gave skew 2.0-7.1 -- every
        # sample landed in the "easy" band.
        sigma = r.uniform(0.33, 0.75)
        shift = s * (r.lognormal(0, sigma, n) - math.exp(sigma ** 2 / 2))
        x = c + shift if kind == "dist_right_skew" else c - shift
    elif kind == "dist_bimodal":
        # parametrise the gap in component-SD units so it maps directly onto the
        # gmm_mode_separation the difficulty grader reads
        comp_sd = s * 0.42
        sep_sd = r.uniform(1.35, 3.40)
        gap = sep_sd * comp_sd
        w = r.uniform(0.32, 0.68)
        k1 = int(n * w)
        x = np.r_[r.normal(c - gap / 2, comp_sd, k1),
                  r.normal(c + gap / 2, comp_sd, n - k1)]
        r.shuffle(x)
    elif kind == "dist_heavy_tail":
        # excess kurtosis of t(df) is 6/(df-4): df in [4.5,7] spans ~12 down to ~2,
        # straddling the grader's 5.0 / 3.0 cuts instead of sitting far above them
        x = c + s * r.standard_t(r.uniform(4.5, 7.0), n) / 1.4
    elif kind == "dist_uniform":
        half = s * r.uniform(1.5, 2.1)
        x = r.uniform(c - half, c + half, n)
    else:
        return None

    return ({"x": [float(v) for v in x]}, [name],
            {"x": _lab(name, unit), "y": "Frequency"}, {"domain": d})


def g_bivariate(r, kind):
    d, xn, xu, xc, xs = pick_var(r)
    _, yn, yu, yc, ys = pick_var(r, d)
    if yn == xn:
        return None
    n = int(r.integers(120, 501))
    x = r.normal(xc, xs, n)
    xz = (x - x.mean()) / (x.std() + 1e-12)

    if kind in ("rel_strong_positive", "rel_strong_negative"):
        # Solve for the noise that produces a *target* correlation rather than
        # hoping a slope/noise pair lands somewhere strong. For y = xz + N(0,sd),
        # r = 1/sqrt(1+sd^2), so sd = sqrt(1/r^2 - 1). Drawing the target
        # uniformly on [0.72,0.97] straddles the grader's 0.90/0.80 cuts.
        target_r = r.uniform(0.72, 0.97)
        sd = math.sqrt(1.0 / target_r ** 2 - 1.0)
        sign = 1.0 if kind == "rel_strong_positive" else -1.0
        y = yc + ys * (sign * xz + r.normal(0, sd, n))
    elif kind == "rel_none":
        y = r.normal(yc, ys, n)
    elif kind == "rel_nonlinear_monotonic":
        x = r.uniform(0.15, 3.4, n)
        form = r.integers(3)
        if form == 0:
            base = np.exp(r.uniform(1.8, 2.9) * x)
        elif form == 1:
            base = x ** r.uniform(4.0, 6.5)
        else:
            base = -1.0 / (x + r.uniform(0.05, 0.2))
        y = yc + ys * (base - base.mean()) / (base.std() + 1e-12) * 1.4
        y = y + r.normal(0, ys * r.uniform(0.04, 0.11), n)
        x = xc + xs * (x - x.mean()) / (x.std() + 1e-12)
    elif kind == "rel_nonlinear_nonmonotonic":
        sign = 1.0 if r.random() < 0.5 else -1.0
        y = yc + ys * sign * (xz ** 2 - 1.0) + r.normal(0, ys * r.uniform(0.18, 0.42), n)
    elif kind == "rel_heteroscedastic":
        x = np.sort(r.uniform(0.2, 3.6, n))
        y = yc + ys * (r.uniform(0.5, 1.1) * x + r.normal(0, 1, n) * (x ** r.uniform(1.8, 2.6)) * 0.55)
        x = xc + xs * (x - x.mean()) / (x.std() + 1e-12)
    else:
        return None

    return ({"x": [float(v) for v in x], "y": [float(v) for v in y]}, [xn, yn],
            {"x": _lab(xn, xu), "y": _lab(yn, yu)}, {"domain": d})


def g_timeseries(r, kind):
    d, yn, yu, yc, ys = pick_var(r)
    n = int(r.integers(48, 241))
    tname, tfn = TIME_AXES[r.integers(len(TIME_AXES))]
    t = tfn(n, r).astype(float)[:n]
    if len(t) < n:
        t = np.arange(n, dtype=float)
    i = np.arange(n)
    noise = lambda k=1.0: r.normal(0, ys * r.uniform(0.10, 0.26) * k, n)  # noqa: E731

    # Each branch below widens the signal-to-noise range so Kendall tau, the
    # changepoint SNR and the robust z-score land across the grader's easy /
    # medium / hard cuts instead of piling up well past the easy threshold.
    if kind in ("ts_increasing", "ts_decreasing"):
        sign = 1.0 if kind == "ts_increasing" else -1.0
        shape = r.integers(3)
        if shape == 0:
            trend = sign * np.linspace(0, r.uniform(1.2, 4.5), n)
        elif shape == 1:
            trend = sign * (np.expm1(np.linspace(0, r.uniform(0.7, 1.8), n)))
        else:
            trend = sign * r.uniform(1.4, 4.0) / (1 + np.exp(-(i - n / 2) / (n / 8)))
        y = yc + ys * trend + noise(r.uniform(0.50, 2.20))
    elif kind == "ts_seasonal":
        period = int(r.choice([4, 6, 12, 12, 24]))
        amp = r.uniform(1.0, 2.8)
        y = yc + ys * amp * np.sin(2 * np.pi * i / period + r.uniform(0, 2 * np.pi))
        if r.random() < 0.45:
            y = y + ys * amp * 0.32 * np.sin(4 * np.pi * i / period)
        y = y + noise(r.uniform(0.30, 1.30))
    elif kind == "ts_changepoint":
        cp = int(n * r.uniform(0.32, 0.68))
        step = ys * r.uniform(1.2, 4.2) * (1 if r.random() < 0.5 else -1)
        y = yc + np.r_[np.zeros(cp), np.full(n - cp, step)] + noise(0.42)
    elif kind == "ts_anomaly_spike":
        y = yc + noise(0.55)
        k = int(r.integers(1, 3))
        pos = r.choice(np.arange(int(0.12 * n), int(0.88 * n)), k, replace=False)
        for p in pos:
            y[p] += ys * r.uniform(2.5, 9.0) * (1 if r.random() < 0.5 else -1)
    else:
        return None

    return ({"t": [float(v) for v in t], "y": [float(v) for v in y]}, [tname, yn],
            {"x": tname, "y": _lab(yn, yu)}, {"domain": d})


def g_grouped(r, kind):
    d, yn, yu, yc, ys = pick_var(r)
    gname, levels = CAT_SETS[r.integers(len(CAT_SETS))]
    k = int(r.integers(2, min(5, len(levels)) + 1))
    levels = list(levels[:k])
    per = int(r.integers(30, 121))

    vals, grps = [], []
    if kind == "grp_significant_diff":
        # spread of group means in within-group SD units; the low end sits just
        # past the epsilon-squared 0.14 gate, the high end is unmistakable
        offs = np.linspace(0, r.uniform(0.9, 3.4), k)
        r.shuffle(offs)
    else:
        offs = np.zeros(k)
    for j, lv in enumerate(levels):
        m = per + int(r.integers(-8, 9))
        vals.append(r.normal(yc + ys * offs[j], ys * r.uniform(0.75, 1.15), m))
        grps += [lv] * m
    v = np.concatenate(vals)
    return ({"group": grps, "value": [float(z) for z in v]}, [gname, yn],
            {"x": gname, "y": _lab(yn, yu)}, {"domain": d})


def g_composition(r, kind):
    gname, levels = CAT_SETS[r.integers(len(CAT_SETS))]
    k = int(r.integers(3, min(6, len(levels)) + 1))
    levels = list(levels[:k])
    n = int(r.integers(160, 701))
    top = r.uniform(0.58, 0.80)
    rest = r.dirichlet(np.ones(k - 1)) * (1 - top)
    p = np.r_[top, rest]
    labs = list(r.choice(levels, size=n, p=p / p.sum()))
    return ({"category": [str(z) for z in labs]}, [gname],
            {"x": gname, "y": "Count"}, {"domain": "general"})


ARITY_GEN = {
    "univariate": g_univariate, "bivariate": g_bivariate,
    "timeseries": g_timeseries, "grouped": g_grouped, "composition": g_composition,
}
VERIFY = {
    "univariate": lambda dd: tax.verify_univariate(dd["x"]),
    "bivariate": lambda dd: tax.verify_bivariate(dd["x"], dd["y"]),
    "timeseries": lambda dd: tax.verify_timeseries(dd["t"], dd["y"]),
    "grouped": lambda dd: tax.verify_grouped(dd["value"], dd["group"]),
    "composition": lambda dd: tax.verify_composition(dd["category"]),
}


def make_one(cls, seed):
    """One accepted sample of `cls`, or None. Label comes from the gate, never the generator."""
    arity = tax.CLASSES[cls]["arity"]
    gen, ver = ARITY_GEN[arity], VERIFY[arity]
    r = np.random.default_rng(seed)
    for _ in range(MAX_TRIES_PER_HIT):
        try:
            made = gen(r, cls)
        except Exception:
            continue
        if made is None:
            continue
        data, cols, axes, meta = made
        try:
            got, st, _why = ver(data)
        except Exception:
            continue
        if got != cls:            # intended class must be the one the test confirms
            continue
        return dict(provenance="synthetic", source_id=f"synth/{cls}/{seed}",
                    source="synthetic", source_url=None, license="CC0-1.0 (generated)",
                    table=f"synthetic_{meta.get('domain', 'general')}",
                    arity=arity, label=cls, stats=st, columns=[str(c) for c in cols],
                    data=data, axis_labels=axes, generator_seed=int(seed),
                    domain=meta.get("domain"))
    return None


def _worker(args):
    cls, seeds = args
    return [c for c in (make_one(cls, s) for s in seeds) if c is not None]


def main():
    real_counts = Counter(json.loads((CACHE / "real_class_counts.json").read_text()))
    need = {}
    for c in tax.CLASS_NAMES:
        gap = max(SYNTH_FLOOR, int(TARGET * BUFFER) - real_counts.get(c, 0))
        if gap > 0:
            need[c] = gap

    total_need = sum(need.values())
    print(f"synthesis targets (floor {SYNTH_FLOOR}/class, top-up to {int(TARGET * BUFFER)}):")
    for c, g in sorted(need.items(), key=lambda kv: -kv[1]):
        print(f"  {c:32s} need {g:>5,}  (real has {real_counts.get(c, 0):,})")
    print(f"  {'TOTAL':32s} need {total_need:>5,}\n")

    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = []
    rng = np.random.default_rng(20260815)
    for c, g in need.items():
        # generous seed budget: the gate rejects a fraction of every generator
        seeds = rng.integers(0, 2 ** 31 - 1, int(g * 2.6))
        for chunk in np.array_split(seeds, max(1, len(seeds) // 90)):
            jobs.append((c, [int(s) for s in chunk]))
    rng.shuffle(jobs)

    got = Counter()
    written = 0
    with open(OUT, "w") as fh, ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        bar = tqdm(as_completed(futs), total=len(futs), unit="batch",
                   desc="synthesising", ncols=100)
        for fu in bar:
            try:
                res = fu.result()
            except Exception:
                continue
            for cand in res:
                if got[cand["label"]] >= need[cand["label"]]:
                    continue
                fh.write(json.dumps(cand) + "\n")
                got[cand["label"]] += 1
                written += 1
            bar.set_postfix(made=written)

    print(f"\nwrote {written:,} verified synthetic candidates -> {OUT}")
    print(f"{'class':32s} {'needed':>8s} {'made':>8s} {'accept':>8s}")
    print("-" * 60)
    for c in tax.CLASS_NAMES:
        if c in need:
            print(f"{c:32s} {need[c]:>8,} {got[c]:>8,} "
                  f"{100.0 * got[c] / need[c]:>7.0f}%")
    print("-" * 60)
    (CACHE / "synth_class_counts.json").write_text(json.dumps(dict(got), indent=2))


if __name__ == "__main__":
    main()
