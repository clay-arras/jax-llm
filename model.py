import jax
import jax.numpy as jnp

from functools import partial
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass(frozen=True)
class AttentionHeadConfig:
    context_size: int
    num_blocks: int
    vocab_size: int
    n_embd: int

    num_heads: int
    head_size: int
    channels: int
    dropout_rate: float = 0.1


@dataclass(frozen=True)
class Module(ABC):
    def __init_subclass__(cls, **kwargs):  # some claude opus magic, idk how this works
        super().__init_subclass__(**kwargs)
        dataclass(frozen=True)(cls)

    @abstractmethod
    def apply(self, *args, **kwargs) -> jax.Array:
        pass


class SelfAttentionHead(Module):
    config: AttentionHeadConfig

    def init(self, key) -> dict:
        k1, k2, k3 = jax.random.split(key, num=3)
        channels = self.config.channels
        head_size = self.config.head_size

        xavier_scl = jnp.sqrt(2.0 / (channels + head_size))
        return {
            "K": jax.random.normal(k1, (channels, head_size)) * xavier_scl,
            "Q": jax.random.normal(k2, (channels, head_size)) * xavier_scl,
            "V": jax.random.normal(k3, (channels, head_size)) * xavier_scl,
        }

    def apply(self, params, x: jax.Array, key) -> jax.Array:
        cf = self.config

        B, T, C = x.shape
        assert C == cf.channels
        k = x @ params["K"]  # (B x T x head_size)
        q = x @ params["Q"]

        dK = cf.head_size
        weights = q @ jnp.matrix_transpose(k) / jnp.sqrt(dK)  # (B x T x T)
        weights = jnp.where(jnp.tril(jnp.full((T, T), 1)) == 1, weights, -jax.numpy.inf)
        weights = jax.nn.softmax(weights, axis=-1)

        drop_mask = jax.random.uniform(key, (B, T, T)) > cf.dropout_rate
        weights = weights * drop_mask / (1 - cf.dropout_rate)

        v = x @ params["V"]  # (B x T x head_size)
        return weights @ v  # (B x T x head_size)


class MultiAttentionHead(Module):
    config: AttentionHeadConfig
    head: SelfAttentionHead

    def init(self, key) -> dict:
        cf = self.config
        k1, *head_keys = jax.random.split(key, num=1 + cf.num_heads)
        xavier_proj = jnp.sqrt(2.0 / (cf.num_heads * cf.head_size + cf.n_embd))
        return {
            "proj": jax.random.normal(k1, (cf.num_heads * cf.head_size, cf.n_embd))
            * xavier_proj,
            "modules": [self.head.init(hk) for hk in head_keys],
        }

    def apply(self, params, x: jax.Array, key) -> jax.Array:
        cf = self.config
        B, T, C = x.shape
        k_drop, *head_keys = jax.random.split(key, num=1 + cf.num_heads)

        out = jnp.concatenate(
            [
                self.head.apply(pr, x, hk)
                for pr, hk in zip(params["modules"], head_keys)
            ],
            axis=-1,
        )  # (B, T, head_size * num_heads)
        drop_mask = jax.random.uniform(k_drop, out.shape) > cf.dropout_rate
        out = out * drop_mask / (1 - cf.dropout_rate)
        out = out @ params["proj"]  # (B, T, n_embd)
        return out


class FeedForward(Module):
    config: AttentionHeadConfig

    def init(self, key) -> dict:
        k1, k2 = jax.random.split(key, num=2)
        n_embd = self.config.n_embd
        return {
            "fc": jax.random.normal(k1, (n_embd, 4 * n_embd))
            * jnp.sqrt(2.0 / (n_embd + 4 * n_embd)),
            "proj": jax.random.normal(k2, (4 * n_embd, n_embd))
            * jnp.sqrt(2.0 / (4 * n_embd + n_embd)),
        }

    def apply(self, params, x: jax.Array, key) -> jax.Array:
        B, T, n_embd = x.shape
        assert self.config.n_embd == n_embd

        x = x @ params["fc"]
        x = jax.nn.gelu(x)
        x = x @ params["proj"]
        drop_mask = jax.random.uniform(key, x.shape) > self.config.dropout_rate
        return x * drop_mask / (1 - self.config.dropout_rate)  # (B, T, n_embd)


class LayerNorm(Module):
    channels: int

    def init(self) -> dict:
        c = self.channels
        return {
            "gamma": jnp.ones((c)),
            "beta": jnp.zeros((c)),
        }

    def apply(self, params, x: jax.Array) -> jax.Array:
        B, T, n_embd = x.shape
        assert self.channels == n_embd
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)

        unwei = (x - mean) / jnp.sqrt(1e-5 + var)
        wei = unwei * params["gamma"] + params["beta"]
        return wei


class TransformerBlock(Module):
    config: AttentionHeadConfig
    ff: FeedForward
    ln: LayerNorm
    head: MultiAttentionHead

    def init(self, key) -> dict:
        k1, k2 = jax.random.split(key, num=2)
        return {
            "ln1": self.ln.init(),
            "head": self.head.init(k1),
            "ln2": self.ln.init(),
            "ff": self.ff.init(k2),
        }

    def apply(
        self, params, x: jax.Array, key
    ) -> jax.Array:  # output shape: (B, T, n_embd)
        k1, k2 = jax.random.split(key, num=2)
        ln = self.ln  # TODO: why do you want to apply layer norm before?
        x = x + self.head.apply(params["head"], ln.apply(params["ln1"], x), k1)
        x = x + self.ff.apply(params["ff"], ln.apply(params["ln2"], x), k2)
        return x


class CategoricalCrossEntropy(Module):
    def apply(self, x: jax.Array, y: jax.Array) -> jax.Array:
        B, T, vocab_size = x.shape
        # y is shape (B, T, vocab_size), we have one example for each [0, T] group
        x = jax.nn.log_softmax(x, axis=-1)

        # TODO: not sure if this is good
        # if t is zero, then it clamps to -1e2 because log(0) is NaN
        clog = lambda t: jnp.where(t, jnp.maximum(jnp.log(t), -1e2), -1e2)
        loss = -(y * clog(x))
        return loss.sum() / (B * T)


class Transformer(Module):
    config: AttentionHeadConfig
    block: TransformerBlock
    ln: LayerNorm
    loss: CategoricalCrossEntropy

    def init(self, key) -> dict:
        cf = self.config
        k1, k2, k3, k4 = jax.random.split(key, num=4)
        block_keys = jax.random.split(k4, num=cf.num_blocks)
        return {
            "tok_embd": jax.random.normal(k1, (cf.vocab_size, cf.n_embd))
            * jnp.sqrt(2.0 / (cf.vocab_size + cf.n_embd)),
            "pos_embd": jax.random.normal(k2, (cf.context_size, cf.n_embd))
            * jnp.sqrt(2.0 / (cf.context_size + cf.n_embd)),  # expect (T, context_size)
            "blocks": [self.block.init(bk) for bk in block_keys],
            "proj": jax.random.normal(k3, (cf.n_embd, cf.vocab_size))
            * jnp.sqrt(2.0 / (cf.n_embd + cf.vocab_size)),
            "ln_last": self.ln.init(),
        }

    @partial(jax.jit, static_argnames=["self"])
    def apply(self, params, x: jax.Array, y: jax.Array, key) -> jax.Array:
        B, T, vocab_size = x.shape
        cf = self.config

        # NOTE: because we're doing matmul instead of lookup, need to one-hot
        # encode matrices
        posits = jnp.eye(T, cf.context_size) @ params["pos_embd"]  # (T, n_embd)
        tokits = x @ params["tok_embd"]  # (B, T, n_embd)
        x = posits + tokits
        assert x.shape == (B, T, cf.n_embd)

        block_keys = jax.random.split(key, num=cf.num_blocks)
        for block_param, bkey in zip(params["blocks"], block_keys):
            x = self.block.apply(block_param, x, bkey)
        x = self.ln.apply(params["ln_last"], x)
        x = x @ params["proj"]
        return self.loss.apply(x, y)
