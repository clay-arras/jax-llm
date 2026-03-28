import pickle
import jax
import jax.numpy as jnp

with open("ckpt/model.pickle", "rb") as f:
    params = pickle.load(f)

# TODO
