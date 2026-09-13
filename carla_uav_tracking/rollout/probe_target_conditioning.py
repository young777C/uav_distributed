"""Tier-0 Probe B (912 plan): does TARGET-CONDITIONED channel reweighting have headroom on our
crop-DINOv2 features, or are we at the resolution/feature ceiling? ZERO training, offline on the
cached features (runs/tah_cache.pt). Decision gate for the FiLM/adapter idea.

Idea: a FiLM-γ modulation is exactly a per-channel diagonal reweighting of the feature. So the ORACLE
diagonal reweighting (Fisher: emphasize channels where target separates from distractors, using true
labels) is the UPPER BOUND of what any target-conditioned FiLM-γ could achieve. We compare, per episode:
  - baseline  : uniform weights → cosine to the oracle target template → pick / margin
  - oracle-diag: Fisher per-channel weights → weighted cosine → pick / margin  (FiLM-γ upper bound)
  - oracle-LDA : full whitened-mean projection → a stretch upper bound for a richer (non-diagonal) adapter
Template = per-episode mean of target feats (oracle memory) — isolates the FEATURE/channel question from
the memory/drift question. In-sample (train=test) → optimistic upper bound: if even this shows ~no gain,
it's a hard ceiling; if it shows a big gain, a learned FiLM has headroom (a real learned version is lower).

  GO   : oracle-diag pick-acc ≫ baseline (e.g. +≥0.08) and margin up  → build the FiLM modulation (Tier 1)
  NO-GO: negligible gain                                              → resolution/feature ceiling → high-res / stronger backbone
"""
import argparse
import numpy as np
import torch


def _feat(fr):
    f = fr["feat"]
    f = f.numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
    return f.astype(np.float64)


def _norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def episode_frames(ep):
    """Yield (feats(n,384) L2-normed, tidx) for frames with a valid target and >=2 candidates."""
    for fr in ep:
        t = int(fr["tidx"])
        f = _feat(fr)
        if t < 0 or t >= f.shape[0] or f.shape[0] < 2:
            continue
        yield _norm(f), t


def fisher_weights(T, D, eps=1e-6):
    """Per-channel Fisher ratio (μ_T−μ_D)²/(varT+varD) → diagonal reweight, normalized to mean 1."""
    mT, mD = T.mean(0), D.mean(0)
    vT, vD = T.var(0), D.var(0)
    w = (mT - mD) ** 2 / (vT + vD + eps)
    w = w / (w.mean() + eps)
    return w


def lda_project(T, D, eps=1e-3):
    """Whitened mean-difference direction (1-D LDA) — a non-diagonal stretch upper bound. Returns a
    scoring direction; we augment features with this projection is overkill, so we just score along it."""
    mT, mD = T.mean(0), D.mean(0)
    Sw = np.cov(np.vstack([T - mT, D - mD]).T) + eps * np.eye(T.shape[1])
    w = np.linalg.solve(Sw, mT - mD)      # Fisher LDA direction
    return w / (np.linalg.norm(w) + 1e-8)


def score_scheme(frames, w=None, template=None):
    """pick-acc + margin under diagonal weight w (None=uniform). template already weighted+normed."""
    picks = margins = hardest = n = 0
    for f, t in frames:
        fw = _norm(f * w) if w is not None else f
        s = fw @ template                                  # (n,) cosine to (weighted) template
        pick = int(s.argmax())
        picks += (pick == t); n += 1
        others = np.delete(s, t)
        margins += float(s[t] - others.max())
        hardest += float(others.max())
    return picks / max(n, 1), margins / max(n, 1), hardest / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="runs/tah_cache.pt")
    ap.add_argument("--max-eps", type=int, default=0, help="0=all")
    a = ap.parse_args()
    data = torch.load(a.cache, weights_only=False, map_location="cpu")
    eps = data["episodes"]
    if a.max_eps:
        eps = eps[:a.max_eps]
    print(f"[probeB] {len(eps)} episodes, d_feat={data['d_feat']}\n")

    agg = {"base": [], "diag": [], "lda": []}
    wu = {"base": [], "diag": [], "lda": []}     # deployable-realistic: fit on first K frames, eval on rest
    K = 30
    tot_frames = 0
    for ep in eps:
        frames = list(episode_frames(ep))
        if len(frames) < 4:
            continue
        # --- deployable variant: fit modulation from first K frames (online-estimable), eval on rest ---
        if len(frames) > K + 4:
            fit, ev = frames[:K], frames[K:]
            Tf = np.vstack([f[t] for (f, t) in fit]); Df = np.vstack([np.delete(f, t, 0) for (f, t) in fit])
            tw = _norm(Tf.mean(0)); wf = fisher_weights(Tf, Df); dwf = lda_project(Tf, Df)
            wu["base"].append(score_scheme(ev, None, tw))
            wu["diag"].append(score_scheme(ev, wf, _norm(tw * wf)))
            pk = mg = nn = 0
            for f, t in ev:
                s = f @ dwf; pk += (int(s.argmax()) == t); nn += 1; mg += float(s[t] - np.delete(s, t).max())
            wu["lda"].append((pk / nn, mg / nn, 0.0, nn))
        # per-frame target / distractor rows for Fisher/LDA stats (oracle labels)
        T = np.vstack([f[t] for (f, t) in frames])
        D = np.vstack([np.delete(f, t, axis=0) for (f, t) in frames])
        templ = _norm(T.mean(0))                                   # oracle per-episode target template
        w = fisher_weights(T, D)
        templ_w = _norm(templ * w)
        d = lda_project(T, D)
        # baseline
        agg["base"].append(score_scheme(frames, None, templ))
        # oracle diagonal (FiLM-γ upper bound)
        agg["diag"].append(score_scheme(frames, w, templ_w))
        # oracle LDA direction: score candidates by projection onto d (non-diagonal stretch bound)
        picks = mg = n = 0
        for f, t in frames:
            s = f @ d
            picks += (int(s.argmax()) == t); n += 1
            mg += float(s[t] - np.delete(s, t).max())
        agg["lda"].append((picks / n, mg / n, 0.0, n))
        tot_frames += len(frames)

    def rep(d, k):
        A = np.array([x[:2] for x in d[k]])   # pick_acc, margin per episode
        return A[:, 0].mean(), A[:, 1].mean()

    bp, bm = rep(agg, "base"); dp, dm = rep(agg, "diag"); lp, lm = rep(agg, "lda")
    print(f"=== IN-SAMPLE ORACLE (optimistic upper bound) ===")
    print(f"scheme            pick_acc   margin    Δpick vs base")
    print(f"-------------------------------------------------------")
    print(f"baseline (uniform)   {bp:.3f}   {bm:+.3f}      —")
    print(f"oracle-diag (FiLM-γ) {dp:.3f}   {dm:+.3f}   {dp-bp:+.3f}")
    print(f"oracle-LDA  (full)   {lp:.3f}   {lm:+.3f}   {lp-bp:+.3f}")
    wbp, _ = rep(wu, "base"); wdp, _ = rep(wu, "diag"); wlp, _ = rep(wu, "lda")
    print(f"\n=== DEPLOYABLE (fit on first {K} frames, eval on rest — online-estimable + generalizing) ===")
    print(f"baseline (uniform)   {wbp:.3f}      —")
    print(f"diag  (FiLM-γ)       {wdp:.3f}   {wdp-wbp:+.3f}")
    print(f"LDA   (full adapter) {wlp:.3f}   {wlp-wbp:+.3f}")
    print(f"\n[in-sample {len(agg['base'])} eps / {tot_frames} frames; deployable {len(wu['base'])} eps]")
    gate = wdp - wbp   # gate on the DEPLOYABLE (honest) diagonal delta, not the optimistic in-sample
    verdict = ("GO — target-conditioned FiLM has headroom (build Tier 1)" if gate >= 0.08 else
               "MARGINAL — small headroom; weigh cost" if gate >= 0.03 else
               "NO-GO — feature/resolution CEILING (→ high-res render / stronger backbone, not modulation)")
    print(f"→ DEPLOYABLE diag Δpick = {gate:+.3f}  ⇒  {verdict}")
    print(f"   (deployable full-adapter Δpick = {wlp-wbp:+.3f}; in-sample oracle upper bounds diag {dp-bp:+.3f} / full {lp-bp:+.3f})")
    print("NOTE: gate is the honest deployable delta (fit first-K, eval rest). In-sample oracle = optimistic ceiling.")


if __name__ == "__main__":
    main()
