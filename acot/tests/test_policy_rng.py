"""1006 §3.5 / G5: per-step policy RNG — arm-independent, history-independent, repeat-salted."""
import torch

from acot.runtime.policy import AcotPolicy


from acot.runtime.server import rng_base


def pol(mode, seed, repeat):
    p = AcotPolicy.__new__(AcotPolicy)
    p.policy_rng = mode
    p.set_episode(rng_base(seed, repeat))
    return p


def draw(p, stream, step):
    p._reseed(stream, step)
    return torch.randn(4)


def test_same_seed_repeat_step_same_noise_regardless_of_history():
    a, b = pol("per_step", 95000, 0), pol("per_step", 95000, 0)
    torch.randn(100)                                   # arm b consumed extra randomness earlier
    assert torch.equal(draw(a, "dit", 42), draw(b, "dit", 42))


def test_streams_steps_repeats_differ():
    a = pol("per_step", 95000, 0)
    x = draw(a, "dit", 42)
    assert not torch.equal(x, draw(a, "ear", 42))
    assert not torch.equal(x, draw(a, "dit", 43))
    assert not torch.equal(x, draw(pol("per_step", 95000, 1), "dit", 42))
    assert not torch.equal(x, draw(pol("per_step", 95001, 0), "dit", 42))


def test_unseeded_does_not_touch_rng():
    p = pol("unseeded", 95000, 0)
    torch.manual_seed(7)
    ref = torch.randn(3)
    torch.manual_seed(7)
    p._reseed("dit", 42)
    assert torch.equal(ref, torch.randn(3))
