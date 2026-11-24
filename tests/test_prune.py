import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Confirmed benign: numpy/Accelerate BLAS on macOS raises spurious divide-by-zero /
# overflow / invalid-value RuntimeWarnings on ordinary, perfectly-finite matmuls.
# Verified by checking logits.data is finite every time the warning fires.
# See check_blas_quirk.py. Silenced here so it doesn't bury real warnings.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from engine.tensor import Tensor, Adam
from nn.mlp import Linear, ReLU, Sequential
from train.dataset.spirals import generate_spiral_dataset
from prune.prune import DynamicPruner, SparseMLP, cubic_sparsity_schedule


def build_model(hidden):
    """2 -> hidden -> hidden -> 3. hidden=128 saturates at 100% acc even at 90%
    sparsity on this task (confirmed empirically) -- there's no capacity for
    sparsity to bind against. hidden=8 is small enough that pruning actually
    costs something, which is what makes the Pareto curve visible at all."""
    return Sequential(
        Linear(2, hidden),
        ReLU(),
        Linear(hidden, hidden),
        ReLU(),
        Linear(hidden, 3),
    )


def train_with_pruning(target_sparsity, criterion="taylor", allow_regrowth=True,
                        seed=123, epochs=250, hidden=8):
    np.random.seed(seed)
    X, y = generate_spiral_dataset(n_samples=1200, n_classes=3, noise=0.123)

    indices = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    X_train, y_train = X[indices[:split]], y[indices[:split]]
    X_val, y_val = X[indices[split:]], y[indices[split:]]

    model = build_model(hidden)
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    pruner = DynamicPruner(model, optimizer, criterion=criterion, allow_regrowth=allow_regrowth)

    batch_size = 32
    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    total_steps = epochs * steps_per_epoch

    # Prune between 15% and 75% of training, leaving final 25% for sparse fine-tuning
    start_step = int(0.15 * total_steps)
    end_step = int(0.75 * total_steps)

    global_step = 0
    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_shuf, y_shuf = X_train[perm], y_train[perm]

        for start in range(0, len(X_train), batch_size):
            xb = Tensor(X_shuf[start:start + batch_size], requires_grad=False)
            yb = y_shuf[start:start + batch_size]

            logits = model(xb)
            loss = logits.softmax_loss(yb)
            assert np.isfinite(loss.data), f"NaN loss at step {global_step}"

            optimizer.zero_grad()
            loss.backward()

            pruner.update_importance(beta=0.9)
            optimizer.step()

            current_target = cubic_sparsity_schedule(
                global_step, start_step, end_step, target_sparsity
            )
            pruner.step(current_target)
            global_step += 1

    # Final validation accuracy
    val_logits = model(Tensor(X_val, requires_grad=False))
    val_acc = float(np.mean(np.argmax(val_logits.data, axis=1) == y_val))
    achieved_sparsity = pruner.current_sparsity()

    # Compile sparse model & verify exact numerical equivalence with dense masked model
    sparse_model = SparseMLP(model)
    sparse_out = sparse_model.forward(X_val)
    assert np.allclose(val_logits.data, sparse_out, atol=1e-7), "Sparse forward mismatch!"

    return model, sparse_model, X_val, val_acc, achieved_sparsity


def benchmark_compute(dense_model, sparse_model, X_sample, repeats=500):
    # Count full dense FLOPs
    total_params = sum(p.data.size for p in dense_model.parameters() if p.data.ndim == 2)
    total_biases = sum(p.data.size for p in dense_model.parameters() if p.data.ndim == 1)
    full_dense_flops = len(X_sample) * (2 * total_params + total_biases)
    sparse_flops = sparse_model.count_flops(batch_size=len(X_sample))

    # Wall-clock timing: raw NumPy dense matmul vs SparseMLP.
    # FIX: previously all three (W,b) pairs pointed at dense_model.layers (no
    # index), so every "layer" in the timing loop was actually the first Linear
    # layer computed three times. dense_model.layers is the Sequential's list
    # of sublayers in order [Linear, ReLU, Linear, ReLU, Linear] -> the three
    # Linear layers are at indices 0, 2, 4.
    linear_layers = [l for l in dense_model.layers if isinstance(l, Linear)]
    assert len(linear_layers) == 3, f"expected 3 Linear layers, got {len(linear_layers)}"
    (W1, b1), (W2, b2), (W3, b3) = (
        (layer.W.data, layer.b.data) for layer in linear_layers
    )

    t0 = time.perf_counter()
    for _ in range(repeats):
        h1 = np.maximum(0.0, X_sample @ W1 + b1)
        h2 = np.maximum(0.0, h1 @ W2 + b2)
        _ = h2 @ W3 + b3
    dense_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = sparse_model.forward(X_sample)
    sparse_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    return {
        "dense_flops": int(full_dense_flops),
        "sparse_flops": int(sparse_flops),
        "flops_reduction_x": round(full_dense_flops / max(sparse_flops, 1), 2),
        "dense_latency_ms": round(dense_ms, 4),
        "sparse_latency_ms": round(sparse_ms, 4),
    }


def run_pareto_sweep(hidden=8, sparsity_levels=None, seeds=None):
    """hidden=8 (not 128): at hidden=128 this network has ~17k prunable weights,
    which is so much more capacity than this 3-class 2D spiral task needs that
    accuracy stays pinned at ~100% up to 90%+ sparsity regardless of pruning
    criterion -- there's no capacity constraint for sparsity to bind against,
    so taylor and magnitude pruning look identical. At hidden=8 the network has
    just enough capacity that sparsity actually costs something, which is what
    makes the taylor-vs-magnitude comparison meaningful. See test_capacity_binding.py
    for the experiment that established this."""
    sparsity_levels = sparsity_levels if sparsity_levels is not None else [0.0, 0.50, 0.75, 0.90, 0.95]
    seeds = seeds if seeds is not None else [10, 50, 100, 123, 200]

    results = {"taylor": [], "magnitude": [], "compute_benchmarks": [], "hidden": hidden}

    for s in sparsity_levels:
        taylor_accs, mag_accs = [], []
        achieved_t, achieved_m = 0.0, 0.0

        for i, seed in enumerate(seeds):
            m_t, sp_t, X_val, acc_t, ach_t = train_with_pruning(
                s, criterion="taylor", allow_regrowth=True, seed=seed, hidden=hidden
            )
            m_m, sp_m, _, acc_m, ach_m = train_with_pruning(
                s, criterion="magnitude", allow_regrowth=False, seed=seed, hidden=hidden
            )
            taylor_accs.append(acc_t * 100.0)
            mag_accs.append(acc_m * 100.0)
            # FIX: previously recorded only the taylor run's achieved sparsity
            # and reused it for the magnitude row too.
            achieved_t, achieved_m = ach_t, ach_m

            # FIX: previously `if seed == seeds` compared an int to the whole
            # list, which is always False, so this never ran. Benchmark once
            # per sparsity level, on the first seed.
            if i == 0:
                bench = benchmark_compute(m_t, sp_t, X_val)
                bench["target_sparsity"] = s
                results["compute_benchmarks"].append(bench)

        results["taylor"].append({
            "target_sparsity": s,
            "achieved_sparsity": round(achieved_t, 4),
            "mean_acc": round(float(np.mean(taylor_accs)), 2),
            "std_acc": round(float(np.std(taylor_accs)), 2),
            "raw_accs": [round(a, 2) for a in taylor_accs],
        })
        results["magnitude"].append({
            "target_sparsity": s,
            "achieved_sparsity": round(achieved_m, 4),
            "mean_acc": round(float(np.mean(mag_accs)), 2),
            "std_acc": round(float(np.std(mag_accs)), 2),
            "raw_accs": [round(a, 2) for a in mag_accs],
        })

        print(f"Sparsity {s*100:4.0f}% (achieved {achieved_t:.3f}/{achieved_m:.3f}) | "
              f"Taylor+Regrow: {np.mean(taylor_accs):.2f}% ± {np.std(taylor_accs):.2f}% | "
              f"Magnitude: {np.mean(mag_accs):.2f}% ± {np.std(mag_accs):.2f}%")

    with open("pareto_results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_pareto(results)
    return results


def plot_pareto(results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Plot 1: Sparsity-Accuracy Pareto Curve with +/- 1 std error bars
    s_t = [r["achieved_sparsity"] * 100 for r in results["taylor"]]
    m_t = [r["mean_acc"] for r in results["taylor"]]
    e_t = [r["std_acc"] for r in results["taylor"]]

    s_m = [r["achieved_sparsity"] * 100 for r in results["magnitude"]]
    m_m = [r["mean_acc"] for r in results["magnitude"]]
    e_m = [r["std_acc"] for r in results["magnitude"]]

    ax1.errorbar(s_t, m_t, yerr=e_t, marker="o", capsize=4, linewidth=2, label="Taylor Saliency + Regrowth")
    ax1.errorbar(s_m, m_m, yerr=e_m, marker="s", linestyle="--", capsize=4, linewidth=2, label="Magnitude Baseline")
    ax1.set_title(f"Sparsity–Accuracy Pareto Curve ({len(results['taylor'][0]['raw_accs'])} Seeds, hidden={results.get('hidden', '?')})")
    ax1.set_xlabel("Achieved Sparsity (%)")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Active FLOPs Reduction
    sparsities = [b["target_sparsity"] * 100 for b in results["compute_benchmarks"]]
    sparse_kflops = [b["sparse_flops"] / 1000.0 for b in results["compute_benchmarks"]]
    dense_kflops = [b["dense_flops"] / 1000.0 for b in results["compute_benchmarks"]]

    ax2.bar(np.array(sparsities) - 1.5, dense_kflops, width=3.0, alpha=0.4, label="Dense FLOPs (kFLOPs)")
    ax2.bar(np.array(sparsities) + 1.5, sparse_kflops, width=3.0, label="SparseMLP Active FLOPs (kFLOPs)")
    ax2.set_title("Real Inference Compute Cost")
    ax2.set_xlabel("Target Sparsity (%)")
    ax2.set_ylabel("Inference FLOPs (kFLOPs)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("part4_pareto_evidence.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_pareto_sweep(hidden=8, sparsity_levels=[0.0, 0.50, 0.75, 0.90, 0.95, 0.98])
