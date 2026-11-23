import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import time
import numpy as np

from nn.mlp import Linear, ReLU

def cubic_sparsity_schedule(step, start_step, end_step, target_sparsity, initial_sparsity=0.0):
    if step < start_step:
        return initial_sparsity
    if step >= end_step:
        return target_sparsity
    progress = (step - start_step) / float(end_step - start_step)
    return target_sparsity + (initial_sparsity - target_sparsity) * ((1.0 - progress) ** 3)


class DynamicPruner:
    """
    Supports:
      - 'taylor': First-order saliency |w * grad| (uses unmasked_grad for regrowth)
      - 'magnitude': Baseline |w|
      - Optional regrowth via RigL-style gradient magnitude on masked entries
    """
    def __init__(self, model, optimizer, criterion="taylor", allow_regrowth=True, regrow_init=1e-3):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.allow_regrowth = allow_regrowth
        self.regrow_init = regrow_init
        self.ema_scores = {}

        # Only prune 2D weight matrices (keep 1D biases dense)
        self.prunable_params = [p for p in model.parameters() if p.data.ndim == 2]
        for p in self.prunable_params:
            self.ema_scores[id(p)] = np.zeros_like(p.data)

    def update_importance(self, beta=0.9):
        """Accumulate smoothed saliency over mini-batches so noisy batch gradients don't mis-prune."""
        for p in self.prunable_params:
            if self.criterion == "magnitude":
                score = np.abs(p.data)
            elif self.criterion == "taylor":
                # Use dense unmasked_grad if available so pruned entries also get scored for regrowth
                g = p.unmasked_grad if p.unmasked_grad is not None else p.grad
                if g is None:
                    continue
                active = np.ones_like(p.data, dtype=bool) if p.mask is None else p.mask
                # For active weights: |w * g|. For pruned weights (w=0): scaled |g| for regrowth eligibility
                score = np.where(active, np.abs(p.data * g), np.abs(g) * self.regrow_init)
            else:
                raise ValueError(f"Unknown criterion: {self.criterion}")

            self.ema_scores[id(p)] = beta * self.ema_scores[id(p)] + (1.0 - beta) * score

    def step(self, target_sparsity):
        """Enforce global hard sparsity budget across all prunable weight matrices."""
        if target_sparsity <= 0.0:
            return

        all_scores = np.concatenate([self.ema_scores[id(p)].ravel() for p in self.prunable_params])
        total_weights = all_scores.size
        k_prune = int(np.floor(target_sparsity * total_weights))

        if k_prune <= 0:
            return
        if k_prune >= total_weights:
            k_prune = total_weights - 1

        # Exact threshold for hard global sparsity budget
        threshold = np.partition(all_scores, k_prune)[k_prune]

        for p in self.prunable_params:
            scores = self.ema_scores[id(p)]
            old_mask = np.ones_like(p.data, dtype=bool) if p.mask is None else p.mask.copy()
            new_mask = scores > threshold

            if not self.allow_regrowth:
                # Monotonic pruning: once pruned, stays pruned
                new_mask = old_mask & new_mask

            # If regrowth occurred, initialize revived weights opposite to unmasked gradient sign
            g = p.unmasked_grad if p.unmasked_grad is not None else p.grad
            if g is not None:
                revive_vals = -self.regrow_init * np.sign(g)
            else:
                revive_vals = 0.0

            # Honest masking via Part 1 optimizer (zeros pruned weights & resets Adam moments)
            self.optimizer.set_mask(p, new_mask, revive_values=revive_vals)

    def current_sparsity(self):
        total = sum(p.data.size for p in self.prunable_params)
        zeros = sum(np.sum(p.data == 0.0) for p in self.prunable_params)
        return zeros / float(total)


# True Sparse-Aware Forward Pass (Real FLOPs & Speedup)
class SparseLinear:
    """
    Compiles a pruned Linear layer into Compressed Sparse Column (CSC) index lists.
    Avoids multiplying or accumulating over pruned (zero) connections.
    """
    def __init__(self, linear_layer):
        W = linear_layer.W.data
        self.b = linear_layer.b.data
        self.in_features, self.out_features = W.shape
        self.nnz = int(np.count_nonzero(W))

        # Store active (non-zero) input indices and weights per output neuron
        self.active_in_idx = []
        self.active_weights = []
        for j in range(self.out_features):
            nz = np.flatnonzero(W[:, j])
            self.active_in_idx.append(nz)
            self.active_weights.append(W[nz, j])

    def forward(self, x):
        # x shape: (batch_size, in_features)
        batch_size = x.shape[0]
        out = np.broadcast_to(self.b, (batch_size, self.out_features)).copy()
        for j in range(self.out_features):
            idx = self.active_in_idx[j]
            if idx.size > 0:
                out[:, j] += x[:, idx] @ self.active_weights[j]
        return out


class SparseMLP:
    """Executes inference using only active non-zero connections."""
    def __init__(self, sequential_model):
        self.layers = []
        for layer in sequential_model.layers:
            if isinstance(layer, Linear):
                self.layers.append(("linear", SparseLinear(layer)))
            elif isinstance(layer, ReLU):
                self.layers.append(("relu", None))

    def forward(self, x):
        for kind, layer in self.layers:
            if kind == "linear":
                x = layer.forward(x)
            elif kind == "relu":
                x = np.maximum(x, 0.0)
        return x

    def count_flops(self, batch_size=1):
        """Exact Multiply-Accumulate (MAC -> 2 FLOPs) count over active connections only."""
        flops = 0
        for kind, layer in self.layers:
            if kind == "linear":
                # 2 FLOPs (1 mul + 1 add) per active connection + 1 add for bias
                flops += batch_size * (2 * layer.nnz + layer.out_features)
        return flops
