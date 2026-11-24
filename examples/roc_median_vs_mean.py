"""Median or mean ensemble? Compare the per-star ROC of the two aggregates on public
TESS 2-min light curves (SPOC; the files are not in the repository).

    roc_median_vs_mean.py run LCDIR OUTDIR [label:sector ...]
    roc_median_vs_mean.py summarise OUTDIR [FIGURE.png]

``run`` builds the ROC of every star-sector in ``public_test_stars.json`` (eclipsing
binaries are excluded), once, for both aggregates, sharing the network passes, and
stores the false and injected candidate tables. ``summarise`` picks, for each star,
the threshold that gives a stated false-positive rate per day, pools the recovery
over stars of a type, and prints a Markdown table and an optional figure.
"""
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
STARS = json.load(open(os.path.join(HERE, "public_test_stars.json")))


def load(lcdir, tic, sector):
    from astropy.io import fits

    from stella.pipeline import FLARE_SAFE_BITMASK

    fn = [f for f in glob.glob(f"{lcdir}/mastDownload/TESS/*/*{tic:016d}*_lc.fits") if f"-s{sector:04d}-" in f]
    with fits.open(fn[0]) as h:
        d = h[1].data
        t, f, e, q = (np.array(d[k], float) for k in ("TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "QUALITY"))
    ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & ((q.astype(int) & FLARE_SAFE_BITMASK) == 0)
    return t[ok], f[ok], e[ok]


def run(lcdir, outdir, only):
    from astropy.table import Table  # noqa: F401

    from stella.vetting import EnsembleScorer, flare_roc

    os.makedirs(outdir, exist_ok=True)
    scorer = EnsembleScorer()
    for s in STARS["stars"]:
        key = f"{s['label']}_s{s['sector']}"
        if only and f"{s['label']}:{s['sector']}" not in only:
            continue
        if os.path.exists(f"{outdir}/{key}.json"):
            continue
        t, f, e = load(lcdir, s["tic"], s["sector"])
        rocs = flare_roc(t, f, e, scorer=scorer, aggregate=("median", "mean"), n_injections=240, seed=0)
        info = dict(s, aggregates={})
        for how, r in rocs.items():
            r.false_table.write(f"{outdir}/{key}_{how}_false.ecsv", overwrite=True)
            r.injections.write(f"{outdir}/{key}_{how}_inj.ecsv", overwrite=True)
            info["aggregates"][how] = dict(exposure_days=r.exposure_days, amplitudes=list(map(float, r.amplitudes)),
                                           meta={k: (list(v) if isinstance(v, tuple) else v) for k, v in r.meta.items()})
        json.dump(info, open(f"{outdir}/{key}.json", "w"))
        print("done", key, flush=True)


def load_rocs(outdir):
    from astropy.table import Table

    from stella.vetting import FlareROC

    out = []
    for fn in sorted(glob.glob(f"{outdir}/*.json")):
        info = json.load(open(fn))
        key = os.path.basename(fn)[:-5]
        rocs = {}
        for how, a in info["aggregates"].items():
            rocs[how] = FlareROC(Table.read(f"{outdir}/{key}_{how}_false.ecsv"), Table.read(f"{outdir}/{key}_{how}_inj.ecsv"),
                                 a["exposure_days"], a["amplitudes"], a["meta"])
        out.append((info, rocs))
    return out


def pooled(entries, how, statistic, target, amplitude=None):
    """Per-star threshold for ``target`` false positives per day; pooled recovery and false-positive rate."""
    from stella.vetting import binomial_interval, poisson_interval

    k = n = nf = 0
    exp = 0.0
    for info, rocs in entries:
        r = rocs[how]
        thr = r.threshold_for_fp_rate(target, statistic)
        rec, _, _, ni = r.recovery(statistic, thr, amplitude)
        k, n = k + int(round(rec * ni)), n + ni
        nf, exp = nf + r.fp_rate(statistic, thr)[3], exp + r.exposure_days
    lo, hi = binomial_interval(k, n)
    flo, fhi = poisson_interval(nf, exp)
    return dict(rec=k / n, lo=float(lo), hi=float(hi), n=n, fp=nf / exp, fp_lo=float(flo), fp_hi=float(fhi), days=exp)


def summarise(outdir, figure=None):
    entries = load_rocs(outdir)
    names = STARS["group_names"]
    groups = [("all", entries)] + [(names[g], [e for e in entries if e[0]["group"] == g]) for g in names]
    print(f"{len(entries)} star-sectors, {sum(e[1]['median'].exposure_days for e in entries):.0f} d scored\n")
    for statistic in ("prob", "combined"):
        print(f"\n### statistic `{statistic}`: pooled recovery of injected flares at the per-star threshold for a stated false-positive rate\n")
        print("| stars (star-sectors, days) | target FP/day | median: recovery (68%) | mean: recovery (68%) | median: realised FP/day | mean: realised FP/day |")
        print("|---|---|---|---|---|---|")
        for gname, ents in groups:
            if not ents:
                continue
            days = sum(e[1]["median"].exposure_days for e in ents)
            for target in (0.01, 0.1):
                m, a = pooled(ents, "median", statistic, target), pooled(ents, "mean", statistic, target)
                fmt = lambda x: f"{x['rec']:.2f} ({x['lo']:.2f}-{x['hi']:.2f})"
                print(f"| {gname} ({len(ents)}, {days:.0f} d) | {target} | {fmt(m)} | {fmt(a)} | {m['fp']:.3f} | {a['fp']:.3f} |")
        print("\nSame, injected amplitude 1% only (all stars):\n")
        for target in (0.01, 0.1):
            m, a = pooled(entries, "median", statistic, target, 0.01), pooled(entries, "mean", statistic, target, 0.01)
            print(f"- target {target}/day: median {m['rec']:.2f} ({m['lo']:.2f}-{m['hi']:.2f}), mean {a['rec']:.2f} ({a['lo']:.2f}-{a['hi']:.2f})")
    if figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        targets = np.geomspace(0.003, 3, 14)
        for ax, statistic in zip(axes, ("prob", "combined")):
            for how, c in (("median", "C0"), ("mean", "C1")):
                pts = [pooled(entries, how, statistic, t) for t in targets]
                x = np.array([max(p["fp"], 1e-3) for p in pts])
                y = np.array([p["rec"] for p in pts])
                lo, hi = np.array([p["lo"] for p in pts]), np.array([p["hi"] for p in pts])
                o = np.argsort(x)
                ax.plot(x[o], y[o], "o-", color=c, label=how, ms=3)
                ax.fill_between(x[o], lo[o], hi[o], color=c, alpha=0.2)
            ax.set_xscale("log")
            ax.set_xlabel("pooled false positives per day (inverted light curves)")
            ax.set_title(f"statistic: {statistic}")
            ax.legend()
        axes[0].set_ylabel("recovery of injected flares")
        fig.tight_layout()
        fig.savefig(figure, dpi=130)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], sys.argv[3], set(sys.argv[4:]))
    else:
        summarise(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
