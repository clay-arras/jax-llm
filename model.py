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
        channels = self.config.channels
        head_size = self.config.head_size
        return {
            "K": jax.random.normal(key, (channels, head_size)),
            "Q": jax.random.normal(key, (channels, head_size)),
            "V": jax.random.normal(key, (channels, head_size)),
        }

    def apply(self, params, x: jax.Array, key) -> jax.Array:
        B, T, C = x.shape
        assert C == self.config.channels
        k = x @ params["K"]  # (B x T x head_size)
        q = x @ params["Q"]

        dK = self.config.head_size
        weights = q @ k.reshape(B, dK, T) / jnp.sqrt(dK)  # (B x T x T)
        weights = jnp.where(jnp.tril(jnp.full((T, T), 1)) == 1, weights, -jax.numpy.inf)
        weights = jax.nn.softmax(weights, axis=-1)

        drop_mask = jax.random.uniform(key, (B, T, T)) > self.config.dropout_rate
        weights = weights * drop_mask

        v = x @ params["V"]  # (B x T x head_size)
        return weights @ v  # (B x T x head_size)


class MultiAttentionHead(Module):
    config: AttentionHeadConfig
    head: SelfAttentionHead

    def init(self, key) -> dict:
        cf = self.config
        return {
            "proj": jax.random.normal(key, (cf.num_heads * cf.head_size, cf.n_embd)),
            "modules": [self.head.init(key) for _ in range(cf.num_heads)],
        }

    def apply(self, params, x: jax.Array, key) -> jax.Array:
        B, T, C = x.shape
        cf = self.config

        out = jnp.concatenate(
            [self.head.apply(pr, x, key) for pr in params["modules"]], axis=-1
        )  # (B, T, head_size * num_heads)
        drop_mask = jax.random.uniform(key, out.shape) > cf.dropout_rate
        out = out * drop_mask
        out = out @ params["proj"]  # (B, T, n_embd)
        return out


class FeedForward(Module):
    config: AttentionHeadConfig

    def init(self, key) -> dict:
        n_embd = self.config.n_embd
        return {
            "fc": jax.random.normal(key, (n_embd, 4 * n_embd)),
            "proj": jax.random.normal(key, (4 * n_embd, n_embd)),
        }

    def apply(self, params, x: jax.Array, key) -> jax.Array:
        B, T, n_embd = x.shape
        assert self.config.n_embd == n_embd

        x = x @ params["fc"]
        x = jax.nn.gelu(x)
        x = x @ params["proj"]
        drop_mask = jax.random.uniform(key, x.shape) > self.config.dropout_rate
        return x * drop_mask  # (B, T, n_embd)


class LayerNorm(Module):
    channels: int

    def init(self, key) -> dict:
        c = self.channels
        return {
            "gamma": jax.random.normal(key, (c)),
            "beta": jax.random.normal(key, (c)),
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
        return {
            "ln1": self.ln.init(key),
            "head": self.head.init(key),
            "ln2": self.ln.init(key),
            "ff": self.ff.init(key),
        }

    def apply(
        self, params, x: jax.Array, key
    ) -> jax.Array:  # output shape: (B, T, n_embd)
        ln = self.ln  # TODO: why do you want to apply layer norm before?
        x = x + self.head.apply(params["head"], ln.apply(params["ln1"], x), key)
        x = x + self.ff.apply(params["ff"], ln.apply(params["ln2"], x), key)
        return x


class BinaryCrossEntropy(Module):
    def apply(self, x: jax.Array, y: jax.Array) -> jax.Array:
        B, T, vocab_size = x.shape
        # y is shape (B, 1, vocab_size), need to broadcast

        # TODO: not sure if this is good
        # if t is zero, then it clamps to -1e2 because log(0) is NaN
        clog = lambda t: jnp.where(t, jnp.maximum(jnp.log(t), -1e2), -1e2)
        loss = -(y * clog(x) + (1 - y) * clog(1 - x))
        return loss.sum()


class Transformer(Module):
    config: AttentionHeadConfig
    block: TransformerBlock
    ln: LayerNorm
    loss: BinaryCrossEntropy

    def init(self, key) -> dict:
        cf = self.config
        return {
            "tok_embd": jax.random.normal(key, (cf.vocab_size, cf.n_embd)),
            "pos_embd": jax.random.normal(
                key, (cf.context_size, cf.n_embd)
            ),  # expect (T, context_size)
            "blocks": [self.block.init(key) for _ in range(cf.num_blocks)],
            "proj": jax.random.normal(key, (cf.n_embd, cf.vocab_size)),
            "ln_last": self.ln.init(key),
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

        for block_param in params["blocks"]:
            x = self.block.apply(block_param, x, key)
        x = self.ln.apply(params["ln_last"], x)
        x = x @ params["proj"]
        x = jax.nn.softmax(
            x, axis=-1
        )  # TODO: consider using log softmax for num stablility
        return self.loss.apply(x, y)
