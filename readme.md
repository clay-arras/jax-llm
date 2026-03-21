
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




- Transpose instead of reshape in attention
- LayerNorm init: gamma=ones, beta=zeros
- Scale dropout by 1/(1-p)
- Replace BCE + softmax with log_softmax + categorical CE
- Split RNG keys in all init methods and fold_in per training step
- Xavier/Glorot weight init scaling
- Add training flag to disable dropout at inference
- Remove inner @jit decorators, keep only outermost
- vmap over heads, lax.scan over blocks when scaling
- Slice position embeddings instead of matmul with eye
