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


def by_star(entries):
    """{star label: [entries of its sectors]}: the sectors of a star are not independent."""
    out = {}
    for e in entries:
        out.setdefault(e[0]["label"], []).append(e)
    return out


def paired(entries, statistic, target, amplitude=None):
    """Per-star recovery(median) - recovery(mean), each at its own per-star threshold for
    ``target`` false positives per day. A star with two sectors counts once (the mean of its sectors)."""
    d = []
    for label, ents in by_star(entries).items():
        x = []
        for info, rocs in ents:
            rr = [rocs[h].recovery(statistic, rocs[h].threshold_for_fp_rate(target, statistic), amplitude)[0]
                  for h in ("median", "mean")]
            x.append(rr[0] - rr[1])
        d.append(np.mean(x))
    return np.array(d)


def cross_validated(entries, how, statistic, target, amplitude=None):
    """Out-of-sample check on the stars observed in two or more sectors: the threshold for ``target``
    false positives per day is chosen on one sector's inverted light curve and applied to another
    sector of the same star, for which the realised false-positive rate and the recovery are measured."""
    from stella.vetting import binomial_interval, poisson_interval

    k = n = nf = 0
    exp = 0.0
    n_pairs = 0
    for label, ents in by_star(entries).items():
        for i, (_, ra) in enumerate(ents):
            for j, (_, rb) in enumerate(ents):
                if i == j:
                    continue
                thr = ra[how].threshold_for_fp_rate(target, statistic)
                rec, _, _, ni = rb[how].recovery(statistic, thr, amplitude)
                k, n = k + int(round(rec * ni)), n + ni
                nf, exp = nf + rb[how].fp_rate(statistic, thr)[3], exp + rb[how].exposure_days
                n_pairs += 1
    if n == 0:
        return None
    lo, hi = binomial_interval(k, n)
    flo, fhi = poisson_interval(nf, exp)
    return dict(rec=k / n, lo=float(lo), hi=float(hi), fp=nf / exp, fp_lo=float(flo), fp_hi=float(fhi),
                days=exp, n_pairs=n_pairs)


def summarise(outdir, figure=None):
    entries = load_rocs(outdir)
    names = STARS["group_names"]
    groups = [("all", entries)] + [(names[g], [e for e in entries if e[0]["group"] == g]) for g in names]
    stars = by_star(entries)
    days = sum(e[1]["median"].exposure_days for e in entries)
    typical = float(np.median([e[1]["median"].exposure_days for e in entries]))
    print(f"{len(entries)} star-sectors of {len(stars)} stars, {days:.0f} d scored; a typical sector has {typical:.0f} d, "
          f"so a target below {1 / typical:.3f}/day means zero false candidates on the star's own inverted curve\n")
    for statistic in ("prob", "combined"):
        print(f"\n### statistic `{statistic}`: pooled recovery of injected flares at the per-star threshold for a stated false-positive rate\n")
        print("The threshold is chosen on each star-sector's own inverted curve, so the realised FP/day is in-sample "
              "(it is 0 by construction when the target is below one false candidate per sector).\n")
        print("| stars (star-sectors, days) | target FP/day | median: recovery (68%) | mean: recovery (68%) | median: realised FP/day | mean: realised FP/day |")
        print("|---|---|---|---|---|---|")
        for gname, ents in groups:
            if not ents:
                continue
            gdays = sum(e[1]["median"].exposure_days for e in ents)
            for target in (0.01, 0.1):
                m, a = pooled(ents, "median", statistic, target), pooled(ents, "mean", statistic, target)
                fmt = lambda x: f"{x['rec']:.2f} ({x['lo']:.2f}-{x['hi']:.2f})"
                print(f"| {gname} ({len(ents)}, {gdays:.0f} d) | {target} | {fmt(m)} | {fmt(a)} | {m['fp']:.3f} | {a['fp']:.3f} |")
        print(f"\nPaired over the {len(stars)} stars (recovery of the median minus recovery of the mean, each at its own per-star "
              "threshold; a star seen in two sectors counts once; mean and standard error over stars, and the number of stars "
              "where the median is higher / lower):\n")
        for target in (0.01, 0.1):
            for amp in (None, 0.01):
                d = paired(entries, statistic, target, amp)
                se = d.std(ddof=1) / np.sqrt(len(d))
                print(f"- target {target}/day, amplitude {'all' if amp is None else amp}: {d.mean():+.3f} +- {se:.3f}"
                      f" ({(d > 0).sum()} up, {(d < 0).sum()} down, {(d == 0).sum()} equal of {len(d)})")
        print("\nSame, injected amplitude 1% only (all stars):\n")
        for target in (0.01, 0.1):
            m, a = pooled(entries, "median", statistic, target, 0.01), pooled(entries, "mean", statistic, target, 0.01)
            print(f"- target {target}/day: median {m['rec']:.2f} ({m['lo']:.2f}-{m['hi']:.2f}), mean {a['rec']:.2f} ({a['lo']:.2f}-{a['hi']:.2f})")
        print("\nOut of sample: stars seen in two sectors, threshold from one sector applied to the other "
              "(false-positive rate and recovery both measured on the other sector):\n")
        print("| target FP/day | aggregate | recovery (68%) | realised FP/day (68%) | days | sector pairs |")
        print("|---|---|---|---|---|---|")
        for target in (0.01, 0.1):
            for how in ("median", "mean"):
                x = cross_validated(entries, how, statistic, target)
                if x:
                    print(f"| {target} | {how} | {x['rec']:.2f} ({x['lo']:.2f}-{x['hi']:.2f}) | "
                          f"{x['fp']:.3f} ({x['fp_lo']:.3f}-{x['fp_hi']:.3f}) | {x['days']:.0f} | {x['n_pairs']} |")
    if figure:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        targets = np.geomspace(0.003, 3, 14)
        for ax, statistic in zip(axes, ("prob", "combined")):
            for how, c in (("median", "C0"), ("mean", "C1")):
                pts = [pooled(entries, how, statistic, t) for t in targets]
                y = np.array([p["rec"] for p in pts])
                lo, hi = np.array([p["lo"] for p in pts]), np.array([p["hi"] for p in pts])
                ax.plot(targets, y, "o-", color=c, label=how, ms=3)
                ax.fill_between(targets, lo, hi, color=c, alpha=0.2)
            ax.axvspan(targets[0], 1 / typical, color="0.85", zorder=0)
            ax.text(targets[0] * 1.1, 0.02, "below one false\ncandidate per sector", fontsize=7, color="0.4")
            ax.set_xscale("log")
            ax.set_xlabel("target false positives per day (per-star thresholds)")
            ax.set_title(f"statistic: {statistic}")
            ax.legend()
        axes[0].set_ylabel("pooled recovery of injected flares")
        fig.tight_layout()
        fig.savefig(figure, dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        run(sys.argv[2], sys.argv[3], set(sys.argv[4:]))
    else:
        summarise(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
