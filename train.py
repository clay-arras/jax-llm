import jax
import jax.numpy as jnp
import pickle


import time
from model import (
    AttentionHeadConfig,
    MultiAttentionHead,
    SelfAttentionHead,
    FeedForward,
    LayerNorm,
    TransformerBlock,
    Transformer,
    CategoricalCrossEntropy,
)

with open("data/data_small.txt", "r", encoding="utf-8") as f:
    raw = f.read()
tok_to_char = list(sorted((set(raw))))
char_to_tok = {v: i for i, v in enumerate(tok_to_char)}
vocab_size = len(tok_to_char)
tok_data = jnp.array([char_to_tok[k] for k in raw])

config = AttentionHeadConfig(
    context_size=64,
    num_blocks=4,
    vocab_size=vocab_size,
    n_embd=128,
    num_heads=4,
    head_size=32,
    channels=128,
    dropout_rate=0.1,
)
B, T = 16, config.context_size  # T == context_size for training
epochs = int(1e4)
learning_rate = 1e-3


slice = lambda data, st, sz: jax.lax.dynamic_slice(data, (st,), (sz,))
pslice = jax.vmap(slice, in_axes=(None, 0, None))


def get_batch(key, data):
    cuts = jax.random.randint(key, (B), 0, len(data) - config.context_size - 1)
    X = pslice(data, cuts, config.context_size)
    X = jax.nn.one_hot(X, num_classes=config.vocab_size)

    y = pslice(data, cuts + 1, config.context_size)
    y = jax.nn.one_hot(y, num_classes=config.vocab_size)
    return X, y


def main() -> None:
    key = jax.random.PRNGKey(42)

    # TODO: gotta figure out better way :p
    heads = MultiAttentionHead(config=config, head=SelfAttentionHead(config=config))
    ff = FeedForward(config=config)
    ln = LayerNorm(channels=config.n_embd)
    block = TransformerBlock(config=config, ff=ff, ln=ln, head=heads)
    gpt = Transformer(config=config, block=block, ln=ln, loss=CategoricalCrossEntropy())

    params = gpt.init(key)
    grad_apply = jax.value_and_grad(gpt.apply)

    st = time.perf_counter()
    for it in range(epochs):
        _, key = jax.random.split(key, num=2)  # refresh key
        X, y = get_batch(key, tok_data)
        loss, grad = grad_apply(params, X, y, key)
        if it % 100 == 0:
            with open(f"ckpt/model-{it}.pickle", "wb") as file:
                pickle.dump(params, file)
            print(loss)
        upd = jax.tree.map(lambda p, g: p - learning_rate * g, params, grad)
        params = upd

    ed = time.perf_counter()
    print(ed - st)


if __name__ == "__main__":
    main()
