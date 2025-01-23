import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import lecun_normal
from flax.linen.linear import DenseGeneral
from typing import Any, Callable, Tuple


def spectral_norm(w, num_iters=1, eps=1e-12):
    def power_iteration(w, u, num_iters):
        for _ in range(num_iters):
            v = jnp.dot(w.T, u)
            v = v / (jnp.linalg.norm(v) + eps)
            u = jnp.dot(w, v)
            u = u / (jnp.linalg.norm(u) + eps)
        return u, v  # type: ignore

    u = jax.random.normal(jax.random.PRNGKey(0), (w.shape[0], 1))
    u, v = power_iteration(w, u, num_iters)
    sigma = jnp.dot(u.T, jnp.dot(w, v))
    return w / sigma


class SNConv3d(nn.Module):
    features: int
    kernel_size: Tuple[int, int, int]
    strides: Tuple[int, int, int] = (1, 1, 1)
    padding: str = "SAME"
    use_bias: bool = True
    num_iters: int = 1

    @nn.compact
    def __call__(self, x):
        kernel = self.param(
            "kernel", lecun_normal(), (self.kernel_size + (x.shape[-1], self.features))
        )
        kernel = spectral_norm(kernel, self.num_iters)
        y = nn.Conv(
            features=self.features,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            kernel_init=lambda *_: kernel,
        )(x)
        return y


class SNLinear(nn.Module):
    features: int
    use_bias: bool = True
    num_iters: int = 1

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", lecun_normal(), (x.shape[-1], self.features))
        kernel = spectral_norm(kernel, self.num_iters)
        y = nn.Dense(
            features=self.features,
            use_bias=self.use_bias,
            kernel_init=lambda *_: kernel,
        )(x)
        return y
