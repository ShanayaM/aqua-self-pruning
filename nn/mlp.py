import numpy as np
from engine.tensor import Tensor


class Module:
    def parameters(self):
        return []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features):
        std = np.sqrt(2.0 / in_features)  # He initialization
        self.W = Tensor(np.random.randn(in_features, out_features) * std, requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x):
        return (x @ self.W.masked()) + self.b

    def parameters(self):
        return [self.W, self.b]


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
