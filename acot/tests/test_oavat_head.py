"""1006 G9: OA-VAT identity adaptation — behaviour with a stub feature extractor (colour inside the box → feature):
reference from the opening pick, no identity check while the locked id is visible, re-acquisition after loss with the
reacq threshold, abstention, REDETECT after max_predict_ticks."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from acot.runtime.head_oavat import OAVAT, OavatHead

pytestmark = pytest.mark.skipif(not __import__("os").path.isdir(OAVAT), reason="external/OA-VAT not cloned")


class StubEx:
    def extract_features(self, bgr, boxes):
        out = []
        for x1, y1, x2, y2 in np.asarray(boxes, int):
            patch = bgr[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)].reshape(-1, 3).astype(float)
            out.append(torch.tensor(patch.mean(0) + 1e-3))
        return torch.stack(out)


def frame(cars):
    img = np.zeros((512, 512, 3), np.uint8)
    for (u, v, col) in cars:
        img[int(v) - 10:int(v) + 10, int(u) - 10:int(u) + 10] = col
    return img


RED, PINK, BLUE = (200, 10, 10), (190, 30, 30), (10, 10, 200)


def D(*cars):
    return [NS(key=k, u=u, v=v, w=20.0, h=20.0) for k, u, v, _ in cars], frame([(u, v, c) for _, u, v, c in cars])


def test_lock_track_lose_reacquire_and_abstain():
    h = OavatHead(StubEx(), match_thresh=0.75, reacq_thresh=0.70, max_predict_ticks=3, enhance_every=0)
    dets, img = D((1, 100, 100, RED), (2, 300, 100, BLUE))
    z = h.score(dets, img, None, init_slot=0)
    assert int(z.argmax()) == 0 and h.locked == 1                        # reference = opening pick
    dets, img = D((1, 110, 100, BLUE), (2, 300, 100, RED))              # locked id now LOOKS blue: no identity check
    assert int(h.score(dets, img, None, init_slot=None).argmax()) == 0
    dets, img = D((2, 300, 100, BLUE))                                  # locked id gone, only a blue car
    z = h.score(dets, img, None, init_slot=None)
    assert z.max() < -1e8 and h.lost == 1                               # blue is not similar enough → abstain
    dets, img = D((2, 300, 100, BLUE), (7, 125, 100, PINK))             # target back with a NEW id, near prediction
    z = h.score(dets, img, None, init_slot=None)
    assert int(z.argmax()) == 1 and h.locked == 7 and h.lost == 0


def test_redetect_after_max_predict():
    h = OavatHead(StubEx(), max_predict_ticks=2, enhance_every=0)
    dets, img = D((1, 100, 100, RED), (2, 300, 100, BLUE))
    h.score(dets, img, None, init_slot=0)
    for _ in range(3):
        h.score(*D((2, 300, 100, BLUE)), None, init_slot=None)
    assert h.redetect
    dets, img = D((2, 300, 100, BLUE), (9, 450, 400, RED))              # far from the prediction: found by REDETECT
    assert int(h.score(dets, img, None, init_slot=None).argmax()) == 1 and h.locked == 9


def test_fresh_after_reset_rebuilds_reference():
    h = OavatHead(StubEx(), enhance_every=0)
    dets, img = D((1, 100, 100, RED), (2, 300, 100, BLUE))
    h.score(dets, img, None, init_slot=0)
    h.reset()
    h.score(dets, img, None, init_slot=1)
    assert h.locked == 2 and len(h.refs) == 1
