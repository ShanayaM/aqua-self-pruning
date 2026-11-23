import numpy as np

class Tensor:
    def __init__(self,
                 inp_data,
                 _previous=(),
                 requires_grad=True,
                 _op: str=""):
        self.grad = None
        self.unmasked_grad = None
        self._previous = set(_previous)
        self._op = _op
        self.requires_grad = requires_grad

        if isinstance(inp_data, np.ndarray):
            self.inp_data = np.array(inp_data.inp_data)
        else:
            pass

    def backward():
        pass

    def softmax_loss():
        pass

    def reduction():
        pass

    def __addition():
        def _backprop():
            pass

    def __subtraction():
        def _backprop():
            pass

    def __multiplication():
        def _backprop():
            pass

    def __division():
        def _backprop():
            pass

    def __relu():
        def _backprop():
            pass

    def __sigmoid():
        def _backprop():
            pass

    def __tanh():
        def _backprop():
            pass

    def reset_zero_grad(self):
        if self.grad is not None:
            self.grad = 0.0

        if self.unmasked_grad is not None:
            self.unmasked_grad = 0.0