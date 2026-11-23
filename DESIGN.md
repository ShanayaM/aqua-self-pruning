# Design choices

### Part 1



### Part 2

Numerical & Training Stability Safeguards (No NaNs)

- Log-Sum-Exp Shift in softmax_cross_entropy: Subtracting np.max(logits, axis=1, keepdims=True) before exp prevents overflow on large logits.

- Per-Parameter Moment Hygiene in Adam: Bias correction (1 - beta1**t, 1 - beta2**t), eps = 1e-8 in the denominator, and moment zeroing whenever a parameter mask changes (pruning or reviving weights) prevent division-by-zero or moment spikes.

- Explicit Finite Assertions: Every epoch asserts np.all(np.isfinite(loss.data)) and checks parameter finiteness across all layers.

