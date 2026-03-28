## todos:

**non-functional**
- consider looking into if loss is done in the apply or update function
- create the update functions
- extra: add path for naming in the init functions

**functional**
- log softmax, and make categorical CE expect logits
- replace scuffed uniform / asint with randint

- add training flag to disable dropout at inference
- vmap over heads, lax.scan over blocks when scaling
- Slice position embeddings instead of matmul with eye

**infrastructure**
- need to standardize benchmarks
- next step: aftter we finish training with batched shakespeare, we can start scaling up and finding a metric to optimize for 
    (iether speed to convergence threshold, or iteration speed)
- todo: need to ingest text data, batch it

---

**benchmarks**
WITHOUT JIT, 153 93.12512896099997
WITH JIT, 178 20.01400013899911


## NEXT TODOS: 

- batch snakespeare, injest and train
- finish the update vs apply function, need to separate inference i.e. softmax
  logits versus cross entropy loss


notes:
- I don't get why T != context_size but the code still works?