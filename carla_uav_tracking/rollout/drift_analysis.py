"""Offline analysis of ASD logs (jsonl from ASD_LOG) — C3 evidence + learned-LLR training.

  python drift_analysis.py auroc  <log.jsonl> [...]            # conf vs ASD stats for detecting a wrong lock
  python drift_analysis.py fit    <train.jsonl> --out llr.json   # learned LLR head (logistic on memory-agnostic stats)
  python drift_analysis.py dyn    <log.jsonl>                   # drift dynamics: wrong-run lengths, absorption

Label per tick: wrong = (mem pick != GT target slot), only on ticks where the target is a visible candidate.
GT is used ONLY offline (analysis / training the LLR head on disjoint seeds), never by the deployed detector.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

NAMES = ["disagree", "margin", "anc_dissim", "lang_disfavor", "disagree_x_margin"]


def load(paths):
    R = []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            r["src"] = p
            R.append(r)
    return R


def labelled(R):
    X, y, c, keys = [], [], [], []
    for r in R:
        if r["x"] is None or r["true"] is None or r["true"] < 0:
            continue
        X.append(r["x"]); y.append(int(r["mem"] != r["true"])); c.append(r["conf"]); keys.append((r["src"], r["ep"]))
    return np.array(X, float), np.array(y), np.array(c, float), keys


def auroc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort() + 1.0
    # tie-correct via average ranks
    from collections import defaultdict
    d = defaultdict(list)
    for v, rk in zip(allv, ranks):
        d[v].append(rk)
    ranks = np.array([np.mean(d[v]) for v in allv])
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(score, y, keys, B=1000, seed=0):
    """Episode-level cluster bootstrap CI of AUROC."""
    rng = np.random.default_rng(seed); eps = sorted(set(keys)); idx = {e: [] for e in eps}
    for i, k in enumerate(keys):
        idx[k].append(i)
    vals = []
    for _ in range(B):
        s = np.concatenate([idx[eps[j]] for j in rng.integers(0, len(eps), len(eps))])
        vals.append(auroc(score[s], y[s]))
    vals = np.array([v for v in vals if v == v])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cmd_auroc(a):
    X, y, c, keys = labelled(load(a.logs))
    print(f"ticks={len(y)}  wrong-rate={y.mean():.3f}  episodes={len(set(keys))}")
    rows = [("memory confidence (−conf; higher=more suspicious)", -c)]
    rows += [(f"ASD stat: {n}", X[:, i]) for i, n in enumerate(NAMES)]
    if a.llr:
        ck = json.load(open(a.llr)); z = (X - np.array(ck["mu"])) / np.array(ck["sd"])
        rows.append(("ASD learned LLR", z @ np.array(ck["w"]) + ck["b"]))
    out = {}
    for name, s in rows:
        lo, hi = boot_ci(s, y, keys)
        v = auroc(s, y); out[name] = [v, lo, hi]
        print(f"  AUROC {v:.3f} [{lo:.3f},{hi:.3f}]  {name}")
    if a.json:
        json.dump({"n_ticks": len(y), "wrong_rate": float(y.mean()), "auroc": out}, open(a.json, "w"), indent=1)


def cmd_fit(a):
    X, y, _, keys = labelled(load(a.logs))
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0; lam = 1e-2
    for _ in range(3000):                                   # L2 logistic regression, full-batch GD
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= 0.1 * (Z.T @ (p - y) / len(y) + lam * w); b -= 0.1 * float((p - y).mean())
    prior = float(np.log(y.mean() / (1 - y.mean())))
    # choose h on TRAINING logs: smallest h with <= a.fa false alarms per 100 correct ticks (CUSUM over wrong-free runs)
    llr = Z @ w + b - prior
    best_h = None
    for h in np.arange(0.5, 12.01, 0.25):
        fa = n_ok = 0
        for k in sorted(set(keys)):
            S = 0.0
            for i in [j for j, kk in enumerate(keys) if kk == k]:
                if y[i] == 0:
                    n_ok += 1
                    S = max(0.0, S + llr[i])
                    if S > h:
                        fa += 1; S = 0.0
                else:
                    S = 0.0
        if n_ok and 100.0 * fa / n_ok <= a.fa:
            best_h = float(h); break
    ck = {"w": w.tolist(), "b": b, "mu": mu.tolist(), "sd": sd.tolist(), "prior_logit": prior,
          "h": best_h if best_h is not None else 6.0, "names": NAMES, "n_train": int(len(y)),
          "train_auroc": auroc(Z @ w + b, y)}
    json.dump(ck, open(a.out, "w"), indent=1)
    print(json.dumps({k: ck[k] for k in ("w", "b", "prior_logit", "h", "n_train", "train_auroc")}))


def cmd_dyn(a):
    R = load(a.logs); eps = {}
    for r in R:
        if r["true"] is None or r["true"] < 0:
            continue
        eps.setdefault((r["src"], r["ep"]), []).append(int(r["mem"] != r["true"]))
    runs, absorbed = [], 0
    for seq in eps.values():
        n = 0
        for v in seq + [0]:
            if v:
                n += 1
            elif n:
                runs.append(n); n = 0
        absorbed += int(len(seq) >= 5 and all(seq[-5:]))
    runs = np.array(runs) if runs else np.zeros(1)
    print(f"episodes={len(eps)} wrong-runs={len(runs)} mean={runs.mean():.1f} ticks p90={np.percentile(runs,90):.1f} "
          f"max={runs.max()} | episodes ending absorbed-in-wrong-lock: {absorbed}/{len(eps)}")


def _episodes(R):
    eps = {}
    for r in R:
        eps.setdefault((r["src"], r["ep"]), []).append(r)
    return eps


def replay(seq, det, beta=0.25, tau=0.1, kappa=0.25, h=2.0, patience=3):
    """Replay one observe-mode episode through a detector variant (open-loop: no correction feedback).
    Returns list of (t_index, alarm_slot). Variants:
      consec   : fire after `patience` consecutive anchor≠mem ticks (the pre-ASD 'anchor' rule)
      cusum1   : single-statistic CUSUM (all disagreements pooled)
      mh       : multi-hypothesis CUSUM (ASD)
    anchor score uses appearance (cosA) + beta*language (zl); beta=0 → appearance-only, 'langonly' → zl only."""
    S1, S, run, alarms = 0.0, {}, 0, []
    for i, r in enumerate(seq):
        if "cosA" not in r:
            continue
        cosA, zl = np.array(r["cosA"]), np.array(r["zl"]); mem = r["mem"]
        a = zl if beta == "langonly" else cosA + beta * zl
        anc = int(a.argmax()); dis = anc != mem; margin = float(a[anc] - a[mem])
        g = (min(margin / tau, 2.0) if dis else 0.0) - kappa
        if det == "consec":
            run = run + 1 if dis else 0
            if run >= patience:
                alarms.append((i, anc)); run = 0
        elif det == "cusum1":
            S1 = max(0.0, S1 + g)
            if S1 > h and dis:
                alarms.append((i, anc)); S1 = 0.0
        else:
            ca = r["aidx"][anc] if dis else None
            for k in list(S):
                if k != ca:
                    S[k] = max(0.0, S[k] - kappa)
            if ca is not None:
                S[ca] = max(0.0, S.get(ca, 0.0) + g)
                if S[ca] > h:
                    alarms.append((i, anc)); S = {}
    return alarms


def detector_scores(eps, **kw):
    """Open-loop detection quality: per wrong-run (≥2 ticks) → detected? delay; alarm points to the TRUE target?
    False alarms = alarms raised while the memory pick is correct (per 100 correct ticks)."""
    n_runs = det = 0; delays = []; fa = n_ok = 0; good = n_al = 0
    for seq in eps.values():
        lab = [None if (r["true"] is None or r["true"] < 0) else int(r["mem"] != r["true"]) for r in seq]
        al = dict(replay(seq, **kw))
        for i, (l, r) in enumerate(zip(lab, seq)):
            if l == 0:
                n_ok += 1; fa += int(i in al)
            if i in al:
                n_al += 1; good += int(r["true"] is not None and al[i] == r["true"])
        i = 0
        while i < len(lab):
            if lab[i] == 1:
                j = i
                while j < len(lab) and lab[j] == 1:
                    j += 1
                if j - i >= 2:
                    n_runs += 1
                    hit = [k for k in range(i, j) if k in al]
                    if hit:
                        det += 1; delays.append(hit[0] - i)
                i = j
            else:
                i += 1
    return {"wrong_runs": n_runs, "detect_rate": det / max(n_runs, 1),
            "delay_ticks": float(np.mean(delays)) if delays else None,
            "fa_per100": 100.0 * fa / max(n_ok, 1), "alarm_correct_slot": good / max(n_al, 1), "alarms": n_al}


def cmd_ablate(a):
    eps = _episodes(load(a.logs))
    variants = [("consec p3 (pre-ASD anchor rule)", dict(det="consec", patience=3)),
                ("consec p5", dict(det="consec", patience=5)),
                ("CUSUM single-stat", dict(det="cusum1")),
                ("ASD multi-hyp CUSUM (app+lang)", dict(det="mh")),
                ("ASD mh, appearance-only (beta=0)", dict(det="mh", beta=0.0)),
                ("ASD mh, language-only", dict(det="mh", beta="langonly")),
                ("ASD mh, beta=0.5", dict(det="mh", beta=0.5))]
    variants += [(f"ASD mh h={h}", dict(det="mh", h=h)) for h in (0.5, 1.0, 4.0, 8.0)]
    out = {}
    for name, kw in variants:
        sc = detector_scores(eps, **kw); out[name] = sc
        print(f"  {name:38s} det={sc['detect_rate']:.2f} delay={sc['delay_ticks']} FA/100={sc['fa_per100']:.2f} "
              f"alarm→true={sc['alarm_correct_slot']:.2f} (runs={sc['wrong_runs']}, alarms={sc['alarms']})")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


def cmd_anchor(a):
    """Anchor quality: fraction of episodes whose anchor window (first k ASD ticks, x=None) locked the true target."""
    eps = _episodes(load(a.logs)); ok = n = 0; per = {}
    for k, seq in eps.items():
        w = [r for r in seq if r["x"] is None and r["true"] is not None and r["true"] >= 0]
        if not w:
            continue
        good = all(r["mem"] == r["true"] for r in w); n += 1; ok += int(good); per[f"{k[0]}#{k[1]}"] = good
    print(f"anchor-correct episodes: {ok}/{n}")
    if a.json:
        json.dump({"anchor_correct": ok, "n": n, "per_episode": per}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("auroc"); p.add_argument("logs", nargs="+"); p.add_argument("--llr"); p.add_argument("--json")
    p = sp.add_parser("fit"); p.add_argument("logs", nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--fa", type=float, default=0.5, help="max false alarms per 100 correct ticks on train logs")
    p = sp.add_parser("dyn"); p.add_argument("logs", nargs="+")
    p = sp.add_parser("ablate"); p.add_argument("logs", nargs="+"); p.add_argument("--json")
    p = sp.add_parser("anchor"); p.add_argument("logs", nargs="+"); p.add_argument("--json")
    a = ap.parse_args()
    {"auroc": cmd_auroc, "fit": cmd_fit, "dyn": cmd_dyn, "ablate": cmd_ablate, "anchor": cmd_anchor}[a.cmd](a)
