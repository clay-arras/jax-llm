
todos:
<!-- - add softmax final layer -->
<!-- - need to flatten the parameters -->
<!-- - add cross entropy loss layer module -->
<!-- - add @abstractmethod with @jax.jit on apply -->

- consider looking into if loss is done in the apply or update function
- create the update functions
- extra: add path for naming in the init functions

WITHOUT JIT, 153 93.12512896099997
WITH JIT, 178 20.01400013899911




<!-- - LayerNorm init: gamma=ones, beta=zeros -->
<!-- - Scale dropout by 1/(1-p) -->
<!-- - Xavier/Glorot weight init scaling (claude) -->
<!-- - Remove inner @jit decorators, keep only outermost -->
<!-- - Split RNG keys in all init methods and fold_in per training step -->
<!-- - Transpose instead of reshape in attention -->
<!-- - Replace BCE + categorical CE -->

- need to standardize benchmarks
- next step: aftter we finish training with batched shakespeare, we can start scaling up and finding a metric to optimize for 
    (iether speed to convergence threshold, or iteration speed)
- todo: need to ingest text data, batch it

<!-- - figure out  -->
<!-- - set up uv and pyproject.toml -->
- log softmax, and make categorical CE expect logits

- Add training flag to disable dropout at inference
- vmap over heads, lax.scan over blocks when scaling
- Slice position embeddings instead of matmul with eye
