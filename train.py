import jax
import jax.numpy as jnp

import time
from model import (
    AttentionHeadConfig,
    MultiAttentionHead,
    SelfAttentionHead,
    FeedForward,
    LayerNorm,
    TransformerBlock,
    Transformer,
    BinaryCrossEntropy,
)


config = AttentionHeadConfig(
    context_size=64,
    num_blocks=4,
    vocab_size=65,
    n_embd=128,
    num_heads=4,
    head_size=32,
    channels=128,
    dropout_rate=0.1,
)
B, T = 16, 8


def main() -> None:
    key = jax.random.PRNGKey(42)
    x = jax.random.uniform(key, (B, T), minval=0, maxval=config.vocab_size).astype(
        jnp.int32
    )
    x = jax.nn.one_hot(x, num_classes=config.vocab_size)

    y = jax.random.uniform(key, (B, 1), minval=0, maxval=config.vocab_size).astype(
        jnp.int32
    )
    y = jax.nn.one_hot(y, num_classes=config.vocab_size)

    # TODO: gotta figure out better way :p
    heads = MultiAttentionHead(config=config, head=SelfAttentionHead(config=config))
    ff = FeedForward(config=config)
    ln = LayerNorm(channels=config.n_embd)
    block = TransformerBlock(config=config, ff=ff, ln=ln, head=heads)
    gpt = Transformer(config=config, block=block, ln=ln, loss=BinaryCrossEntropy())

    params = gpt.init(key)
    grad_apply = jax.value_and_grad(gpt.apply)

    st = time.perf_counter()
    val = 1e100
    it = 0
    while val > 5:
        it += 1
        val, grad = grad_apply(params, x, y, key)
        if it % 10 == 0:
            print(val)
        upd = jax.tree.map(lambda p, g: p - 0.001 * g, params, grad)
        params = upd

    ed = time.perf_counter()
    print(it, ed - st)


if __name__ == "__main__":
    main()
