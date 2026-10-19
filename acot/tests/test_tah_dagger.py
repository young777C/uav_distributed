"""1006 DINOv3 stage 2: the acot runtime's tah_dagger collector — one frame per TAH scoring, in the train/tah_cache.py
schema, with the head's own features (so the backbone of the method is what gets logged)."""
from types import SimpleNamespace as NS

import numpy as np
import torch

from acot.runtime.server import dagger_frame
from acot.runtime.support import COLLECTORS


def _rt(n, d, gt, true):
    head = NS(last=(torch.randn(n, d), torch.rand(n, 3)))
    return NS(policy=NS(head=head), info=NS(gt_idx=tuple(gt), true_idx=true))


def test_schema_and_values():
    rt = _rt(3, 768, (4, -1, 2), 1)
    r = dagger_frame(rt)
    assert r is not None
    last, fr = r
    assert last is rt.policy.head.last
    assert set(fr) == {"feat", "pos", "aidx", "tidx"}
    assert fr["feat"].dtype == np.float16 and fr["feat"].shape == (3, 768)
    assert fr["pos"].dtype == np.float32 and fr["pos"].shape == (3, 3)
    assert fr["aidx"].dtype == np.int16 and fr["aidx"].tolist() == [4, -1, 2]
    assert fr["tidx"] == 1
    np.testing.assert_allclose(fr["feat"], rt.policy.head.last[0].numpy().astype(np.float16))


def test_no_frame_without_scoring():
    rt = _rt(2, 384, (0, -1), 1)
    rt.policy.head.last = None                              # head not scored (reset / no detections)
    assert dagger_frame(rt) is None
    rt = _rt(2, 384, (0, -1, 5), 1)                         # stale scoring: candidate count ≠ this tick's
    assert dagger_frame(rt) is None
    rt = _rt(2, 384, (0, -1), 1)
    rt.info = None
    assert dagger_frame(rt) is None


def test_registered():
    assert "tah_dagger" in COLLECTORS
