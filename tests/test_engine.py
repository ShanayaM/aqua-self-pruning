"""
Run with:  pytest -q test_masking.py   or   python test_masking.py
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from engine.tensor import Tensor, Adam

RNG = np.random.default_rng(123)

# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def numeric_grad(f, x, eps=1e-6):
    """Central-difference d f() / d x.data, perturbing x.data in place."""
    g = np.zeros_like(x.data)
    for i in np.ndindex(*x.data.shape):
        old = x.data[i]
        x.data[i] = old + eps
        fp = float(f().data)
        x.data[i] = old - eps
        fm = float(f().data)
        x.data[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def gradcheck(f, tensors, tol=1e-6):
    for t in tensors:
        t.reset_zero_grad()
    f().backward()
    for t in tensors:
        num = numeric_grad(f, t)
        assert np.allclose(t.grad, num, atol=tol, rtol=1e-5), (t.shape, t.grad, num)


class NaiveAdam:
    """What a quick implementation does: global t, no state reset, updates every
    entry (masked entries merely receive grad == 0). Used to show the failure."""

    def __init__(self, params, lr=0.1, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.reset_zero_grad()

    def step(self):
        self.t += 1
        for k, p in enumerate(self.params):
            g = p.grad
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


def ref_adam_from_scratch(target, n, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Scalar Adam on 0.5*(w-target)^2 starting at w=0 with FRESH state."""
    w = m = v = 0.0
    out = []
    for t in range(1, n + 1):
        g = w - target
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        w -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        out.append(w)
    return np.array(out)


TARGET = np.array([1.0, -2.0, 0.5, 3.0])


def separable_loss(p):
    d = p.masked() - TARGET
    return (d * d).sum() * 0.5          # grad wrt w_eff is (w_eff - TARGET), per entry


def train_step(opt, p):
    opt.zero_grad()
    separable_loss(p).backward()
    opt.step()


# --------------------------------------------------------------------------- #
# 1. the engine itself is correct (ops, broadcasting, shared params)
# --------------------------------------------------------------------------- #
def test_ops_gradcheck():
    x = Tensor(RNG.normal(size=(3, 4)))
    w = Tensor(RNG.normal(size=(4, 5)))
    b = Tensor(RNG.normal(size=(5,)))

    def f():
        z = x @ w + b                                   # broadcasting add
        h = z.tanh() * z.sigmoid()
        d = h / (z * z + 1.0) - 0.5 * z.relu() + (1.0 - z) * 0.1
        return d.mean()

    gradcheck(f, [x, w, b])

    tg = np.array([0, 3, 1])
    gradcheck(lambda: (x @ w + b).softmax_loss(tg), [x, w, b])


def test_shared_parameter_and_dynamic_graph():
    W = Tensor(RNG.normal(size=(4, 3)))
    W.set_mask(RNG.random((4, 3)) > 0.4)
    for n in (2, 5, 3):                                  # graph shape changes per step
        x1, x2 = Tensor(RNG.normal(size=(n, 4))), Tensor(RNG.normal(size=(n, 4)))
        f = lambda: (x1 @ W.masked()).tanh().sum() + ((x2 @ W.masked()) * 0.5).sum()
        gradcheck(f, [W])


# --------------------------------------------------------------------------- #
# 2. forward semantics
# --------------------------------------------------------------------------- #
def test_masked_weight_contributes_zero_to_forward():
    W = Tensor(RNG.normal(size=(4, 3)))
    W.set_mask(RNG.random((4, 3)) > 0.5)
    x = Tensor(RNG.normal(size=(6, 4)), requires_grad=False)
    before = (x @ W.masked()).data.copy()
    W.data[~W.mask] = 1e6                                # garbage under the mask
    after = (x @ W.masked()).data
    assert np.array_equal(before, after)


# --------------------------------------------------------------------------- #
# 3. what the gradient of a masked weight IS
# --------------------------------------------------------------------------- #
def _masked_layer():
    W = Tensor(RNG.normal(size=(4, 3)))
    W.set_mask(RNG.random((4, 3)) > 0.5)
    assert (~W.mask).any() and W.mask.any()
    x = Tensor(RNG.normal(size=(6, 4)), requires_grad=False)
    f = lambda: (x @ W.masked()).tanh().sum()
    return W, f


def test_grad_is_exactly_zero_on_masked_and_correct_on_active():
    W, f = _masked_layer()
    f().backward()
    assert np.all(W.grad[~W.mask] == 0.0)                # exact, not approximate
    num = numeric_grad(f, W)                             # d loss / d stored weight
    assert np.allclose(W.grad, num, atol=1e-7)           # also 0 on masked entries
    # dense gradient agrees with the masked one wherever the weight is active
    assert np.allclose(W.unmasked_grad[W.mask], W.grad[W.mask])
    assert np.any(W.unmasked_grad[~W.mask] != 0.0)       # but is informative elsewhere


def test_unmasked_grad_is_revival_sensitivity():
    """unmasked_grad[i,j] == d loss / d(value of connection i,j) if it were switched
    on at 0. Verified by actually switching it on at +-eps."""
    W, f = _masked_layer()
    f().backward()
    eps = 1e-6
    for i, j in zip(*np.where(~W.mask)):
        base_mask = W.mask.copy()
        vals = []
        for s in (+1, -1):
            W.mask[i, j] = True
            W.data[i, j] = s * eps
            vals.append(float(f().data))
        W.mask[:] = base_mask
        W.data[i, j] = 0.0
        assert abs((vals[0] - vals[1]) / (2 * eps) - W.unmasked_grad[i, j]) < 1e-6


def test_mask_is_snapshotted_at_forward_time():
    W, f = _masked_layer()
    expected_mask = W.mask.copy()
    loss = f()
    W.mask[:] = True                                     # in-place edit between fwd and bwd
    loss.backward()
    assert np.all(W.grad[~expected_mask] == 0.0)


# --------------------------------------------------------------------------- #
# 4. optimizer: pruned weights must not move, even with momentum history
# --------------------------------------------------------------------------- #
def _prime_and_prune(opt_cls, **kw):
    p = Tensor(RNG.normal(size=4))
    opt = opt_cls([p], **kw)
    for _ in range(10):                                  # build real momentum / moments
        train_step(opt, p)
    mask = np.array([True, False, True, True])           # prune entry 1
    return p, opt, mask


def test_pruned_entries_stay_frozen_adam_vs_naive():
    p, opt, mask = _prime_and_prune(Adam, lr=0.1, weight_decay=0.01)
    assert opt.state[id(p)]["m"][1] != 0.0               # there IS history to corrupt
    opt.set_mask(p, mask)
    for _ in range(5):
        train_step(opt, p)
        assert p.data[1] == 0.0 and p.grad[1] == 0.0
        for arr in opt.state[id(p)].values():
            assert arr[1] == 0.0

    # contrast: identical history, same (perfectly zero) masked gradient,
    # but the naive optimizer drags the "frozen" weight around via stale momentum.
    q = Tensor(p.data.copy())
    nopt = NaiveAdam([q], lr=0.1)
    for _ in range(10):
        nopt.zero_grad(); separable_loss(q).backward(); nopt.step()
    q.set_mask(mask)
    for _ in range(5):
        nopt.zero_grad(); separable_loss(q).backward(); nopt.step()
    assert q.grad[1] == 0.0 and q.data[1] != 0.0         # <- the classic bug


def test_direct_mask_edit_is_reconciled_at_step():
    p, opt, mask = _prime_and_prune(Adam, lr=0.1)
    p.set_mask(mask)                                     # no optimizer.set_mask call
    train_step(opt, p)
    assert p.data[1] == 0.0 and opt.state[id(p)]["m"][1] == 0.0


# --------------------------------------------------------------------------- #
# 5. optimizer: revived connections start clean
# --------------------------------------------------------------------------- #
def test_revived_entry_matches_fresh_adam_exactly():
    lr, k = 0.1, 6
    p, opt, mask = _prime_and_prune(Adam, lr=lr)
    opt.set_mask(p, mask)
    for _ in range(3):
        train_step(opt, p)
    opt.set_mask(p, np.ones(4, bool))                    # revive at 0
    assert p.data[1] == 0.0 and all(a[1] == 0.0 for a in opt.state[id(p)].values())
    traj = []
    for _ in range(k):
        train_step(opt, p)
        traj.append(p.data[1])
    ref = ref_adam_from_scratch(TARGET[1], k, lr)
    assert np.allclose(traj, ref, rtol=0, atol=1e-12)
    # per-element bias correction => first step is ~lr, not ~3.16*lr
    assert abs(abs(traj[0]) - lr) < 1e-6

    # contrast: stale state + global step count gives a different, wrong trajectory
    q = Tensor(RNG.normal(size=4)); nopt = NaiveAdam([q], lr=lr)
    for _ in range(10):
        nopt.zero_grad(); separable_loss(q).backward(); nopt.step()
    q.set_mask(mask)
    for _ in range(3):
        nopt.zero_grad(); separable_loss(q).backward(); nopt.step()
    q.set_mask(np.ones(4, bool)); q.data[1] = 0.0        # even if we zero the value by hand
    ntraj = []
    for _ in range(k):
        nopt.zero_grad(); separable_loss(q).backward(); nopt.step()
        ntraj.append(q.data[1])
    assert not np.allclose(ntraj, ref, atol=1e-3)


def test_revive_values_and_untouched_neighbours():
    p, opt, mask = _prime_and_prune(Adam, lr=0.1)
    opt.set_mask(p, mask)
    neighbour_state = {k: v[[0, 2, 3]].copy() for k, v in opt.state[id(p)].items()}
    opt.set_mask(p, np.ones(4, bool), revive_values=0.5)
    assert p.data[1] == 0.5
    for k, v in opt.state[id(p)].items():                # transition must not disturb others
        assert np.array_equal(v[[0, 2, 3]], neighbour_state[k])


# --------------------------------------------------------------------------- #
# 6. end-to-end: dynamic sparse training with prune + regrow by unmasked_grad
# --------------------------------------------------------------------------- #
def test_dynamic_sparse_training_end_to_end():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(128, 10))
    y = (X @ rng.normal(size=(10, 3))).argmax(1)
    W1, b1 = Tensor(rng.normal(size=(10, 24)) * 0.3), Tensor(np.zeros(24))
    W2, b2 = Tensor(rng.normal(size=(24, 3)) * 0.3), Tensor(np.zeros(3))
    for W in (W1, W2):
        W.set_mask(rng.random(W.shape) > 0.5)
    opt = Adam([W1, b1, W2, b2], lr=0.02)
    Xt = Tensor(X, requires_grad=False)

    def forward():
        return ((Xt @ W1.masked() + b1).relu() @ W2.masked() + b2).softmax_loss(y)

    first = float(forward().data)
    for step in range(300):
        opt.zero_grad()
        loss = forward(); loss.backward(); opt.step()
        if step % 20 == 19:                              # rewire: prune k smallest, regrow k best
            for W in (W1, W2):
                act, off = np.flatnonzero(W.mask), np.flatnonzero(~W.mask)
                k = max(1, len(act) // 10)
                drop = act[np.argsort(np.abs(W.data.ravel()[act]))[:k]]
                grow = off[np.argsort(-np.abs(W.unmasked_grad.ravel()[off]))[:k]]
                new = W.mask.ravel().copy(); new[drop] = False; new[grow] = True
                opt.set_mask(W, new.reshape(W.shape))
        for W in (W1, W2):                               # invariants hold at every step
            assert np.all(W.data[~W.mask] == 0.0)
            for arr in opt.state[id(W)].values():
                assert np.all(arr[~W.mask] == 0.0)
    last = float(forward().data)
    assert np.isfinite(last) and last < 0.6 * first, (first, last)


if __name__ == "__main__":
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_")]
    for n, f in tests:
        f()
        print("PASS", n)
    print(f"\n{len(tests)} tests passed")