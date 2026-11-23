import numpy as np

def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape` (undo numpy broadcasting)."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for ax, s in enumerate(shape):
        if s == 1 and grad.shape[ax] != 1:
            grad = grad.sum(axis=ax, keepdims=True)
    return grad

def _acc(t, g):
    """Accumulate gradient g into tensor t (if needed)."""
    if not t.requires_grad:
        return
    t.grad = np.array(g, dtype=np.float64) if t.grad is None else t.grad + g


def _wrap(x):
    return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)


def _make(data, parents, op):
    rg = any(p.requires_grad for p in parents)
    return Tensor(data, _previous=parents, requires_grad=rg, _op=op)


class Tensor:
    def __init__(self,
                 inp_data,
                 _previous=(),
                 requires_grad=True,
                 _op: str=""):
        if isinstance(inp_data, Tensor):
                    inp_data = inp_data.data
        self.data = np.array(inp_data, dtype=np.float64)  # make a private copy
        self.grad = None # dL/dW  (masked: exactly 0 on masked entries)
        self.unmasked_grad = None # dL/dW_eff (dense; only set for masked params)
        self.mask = None # None = dense, else bool array same shape as data
        self._previous = set(_previous)
        self._op = _op
        self.requires_grad = requires_grad
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("backward() on a tensor that does not require grad")
        topo, seen, stack = [], set(), [(self, False)]
        while stack:  # iterative DFS: no recursion limit on deep graphs
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._previous:
                if id(p) not in seen:
                    stack.append((p, False))
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar roots")
            grad = np.ones_like(self.data)
        _acc(self, grad)
        for node in reversed(topo):
            if node.grad is not None:
                node._backward()

    def softmax_loss(self, targets):
        targets = np.asarray(targets, dtype=int)
        z = self.data - self.data.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        n = self.data.shape[0]
        loss = -np.log(p[np.arange(n), targets] + 1e-300).mean()
        out = _make(loss, (self,), "softmax_loss")

        def _backprop():
            d = p.copy()
            d[np.arange(n), targets] -= 1.0
            _acc(self, out.grad * d / n)

        out._backward = _backprop
        return out

    def set_mask(self, mask):
            """Change the mask. Prefer Optimizer.set_mask so optimizer state is
            reconciled immediately (it is also reconciled lazily at step())."""
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != self.data.shape:
                raise ValueError(f"mask shape {mask.shape} != data shape {self.data.shape}")
            self.mask = mask.copy()
    
    def masked(self):
        """Graph node for W_eff = W * M. The mask is SNAPSHOTTED here, so a mask
        edit between forward and backward cannot desynchronise the two passes."""
        if self.mask is None:
            return self
        m = self.mask.copy()
        out = _make(self.data * m, (self,), "mask")

        def _backprop():
            g = out.grad
            if not self.requires_grad:
                return
            _acc(self, g * m)  # true chain rule: exactly 0 where masked
            self.unmasked_grad = g.copy() if self.unmasked_grad is None \
                else self.unmasked_grad + g

        out._backward = _backprop
        return out

    def __addition__(self, oth):
        oth = _wrap(oth)
        out = _make(self.data + oth.data, (self, oth), "+")

        def _backprop():
            _acc(self, _unbroadcast(out.grad, self.data.shape))
            _acc(oth, _unbroadcast(out.grad, oth.data.shape))

        out._backward = _backprop
        return out

    def __subtraction__(self, oth):
        oth = _wrap(oth)
        out = _make(self.data - oth.data, (self, oth), "-")

        def _backprop():
            _acc(self, _unbroadcast(out.grad, self.data.shape))
            _acc(oth, _unbroadcast(-out.grad, oth.data.shape))

        out._backward = _backprop
        return out

    def __multiplication__(self, oth):
        oth = _wrap(oth)
        out = _make(self.data * oth.data, (self, oth), "*")

        def _backprop():
            _acc(self, _unbroadcast(out.grad * oth.data, self.data.shape))
            _acc(oth, _unbroadcast(out.grad * self.data, oth.data.shape))

        out._backward = _backprop
        return out

    def __division__(self, oth):
        oth = _wrap(oth)
        out = _make(self.data / oth.data, (self, oth), "/")

        def _backprop():
            _acc(self, _unbroadcast(out.grad / oth.data, self.data.shape))
            _acc(oth, _unbroadcast(-out.grad * self.data / oth.data ** 2, oth.data.shape))

        out._backward = _backprop
        return out

    def relu(self):
        out = _make(np.maximum(self.data, 0.0), (self,), "relu")
        def _backprop():
            _acc(self, out.grad * (self.data > 0))

        out._backward = _backprop
        return out

    def sigmoid(self):
        s = 1. / (1. + np.exp(-self.data))
        out = _make(s, (self,), 'sigmoid')
        def _backprop():
            _acc(self, out.grad * s * (1. - s))

        out._backward = _backprop
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = _make(t, (self,), 'tanh')

        def _backprop():
            _acc(self, out.grad * (1. - t**2))
        out._backward = _backprop
        return out

    def __matmul__(self, other):
        other = _wrap(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("matmul supports 2-D @ 2-D only")
        out = _make(self.data @ other.data, (self, other), "@")

        def _backprop():
            _acc(self, out.grad @ other.data.T)
            _acc(other, self.data.T @ out.grad)

        out._backward = _backprop
        return out

    def reset_zero_grad(self):
        if self.grad is not None:
            self.grad = None

        if self.unmasked_grad is not None:
            self.unmasked_grad = None

    def reduction(self, axis=None, keepdims=False):
        out = _make(self.data.sum(axis=axis, keepdims=keepdims), (self,), 'sum')
        def _backprop():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis)
            _acc(self, np.broadcast_to(grad, self.data.shape))
        out._backward = _backprop
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod(
            [self.data.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    __radd__ = __addition__
    __rmul__ = __multiplication__
    __add__ = __addition__
    __sub__ = __subtraction__
    __mul__ = __multiplication__
    __truediv__ = __division__

    sum = reduction
    
    def __rsubtraction__(self, other):
        return _wrap(other) - self

    def __rdivision__(self, other):
        return _wrap(other) / self

    __rsub__ = __rsubtraction__
    __rtruediv__ = __rdivision__

    def __neg__(self):
        return self * -1.0

class _MaskAwareOptimizer:
    state_keys = ()

    def __init__(self, params):
        self.params = list(params)
        self.state = {id(p): {k: np.zeros_like(p.data) for k in self.state_keys}
                      for p in self.params}
        self._seen = {}
        for p in self.params:
            self._seen[id(p)] = self._current(p)
            p.data[~self._seen[id(p)]] = 0.0  # invariant: masked weights are 0

    @staticmethod
    def _current(p):
        return np.ones(p.data.shape, bool) if p.mask is None else p.mask.copy()

    def set_mask(self, p, new_mask, revive_values=0.0):
        p.set_mask(new_mask)
        self._sync(p, revive_values)

    def _sync(self, p, revive_values=0.0):
        """Single place where prune/revive transitions are applied."""
        old, new = self._seen[id(p)], self._current(p)
        pruned, revived = old & ~new, ~old & new
        changed = pruned | revived
        if changed.any():
            for arr in self.state[id(p)].values():
                arr[changed] = 0.0  # no stale moments
            p.data[pruned] = 0.0 # frozen to zero
            p.data[revived] = np.broadcast_to(revive_values, p.data.shape)[revived]
        self._seen[id(p)] = new

    def zero_grad(self):
        for p in self.params:
            p.reset_zero_grad()

    def step(self):
        for p in self.params:
            self._sync(p)  # catches mask edits made directly on the tensor
            if p.grad is None:
                continue
            active = self._seen[id(p)]
            if active.any():
                self._update(p, p.grad[active], active, self.state[id(p)])

    def _update(self, p, g, active, st):
        raise NotImplementedError


class Adam(_MaskAwareOptimizer):
    state_keys = ("m", "v", "t")

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay

    def _update(self, p, g, active, st):
        m = self.b1 * st["m"][active] + (1 - self.b1) * g
        v = self.b2 * st["v"][active] + (1 - self.b2) * g * g
        t = st["t"][active] + 1.0          # per-element step count
        st["m"][active], st["v"][active], st["t"][active] = m, v, t
        mhat = m / (1 - self.b1 ** t)
        vhat = v / (1 - self.b2 ** t)
        w = p.data[active]
        w = w - self.lr * self.wd * w      # decoupled decay, active entries only
        p.data[active] = w - self.lr * mhat / (np.sqrt(vhat) + self.eps)

def __init__():
    pass
