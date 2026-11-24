"""A tiny scalar autograd engine (in the spirit of Karpathy's micrograd), fully commented.

Every Value remembers (1) its data, (2) the Values it was computed from, and
(3) a closure `_backward` that pushes its gradient to those parents using the
LOCAL derivative of the operation (the chain rule, one edge at a time).
"""
import math
import random


# --8<-- [start:value]
class Value:
    def __init__(self, data, _parents=(), _op=""):
        self.data = float(data)
        self.grad = 0.0                 # dL/d(self), filled in by backward()
        self._backward = lambda: None   # how to push self.grad to the parents
        self._prev = set(_parents)
        self._op = _op                  # for debugging / drawing the graph

    # ----- operations: each defines forward value AND local backward rule -----
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():                # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():                # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):               # k is a plain number
        assert isinstance(k, (int, float))
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():                # d(a^k)/da = k a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward():                # d(e^a)/da = e^a
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():                # d(ln a)/da = 1/a
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():                # d tanh(a)/da = 1 - tanh(a)^2
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward():                # 1 if a > 0 else 0
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    # ----- derived operations reuse the primitives above -----
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)
    def __radd__(self, other): return self + other
    def __rsub__(self, other): return Value(other) - self
    def __rmul__(self, other): return self * other
    def __rtruediv__(self, other): return Value(other) / self

    # ----- reverse-mode autodiff -----
    def backward(self):
        # 1) topological order: every node appears after all of its parents
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        # 2) seed dL/dL = 1, then apply the chain rule in reverse order
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
# --8<-- [end:value]


# --8<-- [start:nn]
class Neuron:
    def __init__(self, n_in, nonlin=True):
        # uniform init in [-1/sqrt(n_in), 1/sqrt(n_in)] keeps pre-activations O(1)
        bound = 1.0 / math.sqrt(n_in)
        self.w = [Value(random.uniform(-bound, bound)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)   # w·x + b
        return act.tanh() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, n_in, n_out, **kw):
        self.neurons = [Neuron(n_in, **kw) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, n_in, sizes):
        dims = [n_in] + sizes
        self.layers = [Layer(dims[i], dims[i + 1], nonlin=(i != len(sizes) - 1))
                       for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]
# --8<-- [end:nn]


# --8<-- [start:train]
def train_demo(seed=0, steps=200, lr=0.1, verbose=True):
    random.seed(seed)
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]                     # targets
    model = MLP(3, [4, 4, 1])
    for step in range(steps):
        preds = [model(x).tanh() for x in xs]       # squash output to (-1, 1)
        loss = sum(((p - y) ** 2 for p, y in zip(preds, ys)), Value(0.0)) / len(xs)
        for p in model.parameters():                # 1) zero grads
            p.grad = 0.0
        loss.backward()                             # 2) backprop
        for p in model.parameters():                # 3) gradient descent step
            p.data -= lr * p.grad
        if verbose and step % 50 == 0:
            print(f"step {step:3d}  loss {loss.data:.5f}")
    return loss.data, [round(p.data, 3) for p in preds]
# --8<-- [end:train]


if __name__ == "__main__":
    final_loss, preds = train_demo()
    print("final loss", final_loss, "predictions", preds)
