# Vector language reference

Every function below is demonstrated with verified output in [examples.md](examples.md).

## Syntax

| Form | Meaning |
| ---- | ------- |
| `x = expr` | binding (immutable; a new binding shadows) |
| `function name(a, b):` | function — body is an indented expression list, last expression is the value |
| `module Name(arg):` | module — fields (`w = ...`) plus methods; must define `forward(self, ...)` |
| `for i in 0..n:` | loop, compiled to one XLA while op; `by` sets the step: `for x in 0.0..1.0 by 0.25:` |
| `while cond:` | loop until a scalar comparison turns false, compiled to one XLA while op; the body must reassign a binding the condition depends on |
| `x[i]`, `x[iv]` | index rows along axis 0 — a runtime scalar picks one row, a vector gathers rows (same as `take`); negative literals count from the end: `x[-1]` |
| `x[a:b]` | slice along axis 0 with compile-time bounds; open ends and negative bounds work: `x[:3]`, `x[-2:]`; a runtime start with a static width works: `x[i : i + 4]`, and a vector of starts gathers windows: `data[starts : starts + size]` |
| `import mlp` | bring `function`/`module` declarations from `mlp.vec` (dots are subdirectories, path relative to the importing file) |
| `[1.0, 2.0]` | array literal |
| `{a: x, b: y}` | record literal; `.a` accesses a field |
| `< > <= >= == !=` | comparisons, producing booleans for `where` |
| `# comment` | comment to end of line |

Numbers are `f32` by default. Broadcasting aligns trailing dimensions and never stretches size-1 dimensions.

## Transforms

- `grad(f, args...)` — gradient of scalar-valued `f` with respect to its first argument; works on records and module instances (`grad(model.loss, x, t)` returns a model-shaped gradient)
- `vmap(f, args...)` — map `f` over axis 0 of the tensor arguments; `f` may be a builtin (`vmap(matmul, a, b)` is batched matmul) or a method (`vmap(self.sequence_loss, ids, targets)` — the instance passes through unmapped); record arguments pass through unmapped (weights, config), so `vmap(step, model, xb)` maps the batch while the model is shared; nestable, and inner maps lift shallower tensors automatically
- `jacobian(f, x)` — jacobian of vector-valued `f`

## Math

- elementwise: `exp(x)`, `log(x)`, `tanh(x)`, `sqrt(x)`, `sin(x)`, `cos(x)`, `floor(x)`, `abs(x)`, `mod(a, b)`, `pow(a, b)`, `maximum(a, b)`, `minimum(a, b)`
- reductions: `sum(x)`, `mean(x)`, `max(x)`, `min(x)` — optional trailing axis: `sum(m, 0)`
- linear algebra: `matmul(a, b)` — leading dimensions batch like torch (`[h, t, k] @ [h, k, n]`, and `[b, t, k] @ [k, n]` broadcasts); `transpose(m)` — or with an axis permutation: `transpose(x, 1, 0, 2)`
- `softmax(x)` — along the last axis, any rank (a matrix softmaxes per row)
- casts: `f32(x)`, `f64(x)`

## Arrays

- `arange(stop)` / `arange(start, stop)` / `arange(start, stop, step)`
- `linspace(start, stop, count)`
- `zeros(dims...)`, `randn(dims...)`
- `reshape(x, dims...)` — dims are compile-time constants; arithmetic on constants works: `reshape(x, h * w)`
- `slice(x, start, size)` — axis 0; `start` may be a runtime scalar, `size` is static; a vector of starts gathers a batch of windows in one op: `slice(data, starts, size)` → `[count, size, ...]`
- `len(x)` — the leading dimension as a number (fixed at compile time)
- `concat(a, b, ...)` — join along axis 0; `stack(a, b, ...)` — join along a new axis 0; both work under `vmap` (unbatched parts broadcast, like prepending a cls token)
- `take(values, indices)` — fancy indexing along axis 0 (embedding lookups scale to any vocab); differentiable, duplicate indices accumulate gradient, out-of-range indices clamp
- `sort(x)`, `argsort(x)` — vectors; batch with `vmap`; `sort` is differentiable
- `argmax(x)`, `argmin(x)` — first index on ties
- `cumsum(x)`
- `one_hot(indices, depth)`
- `bincount(values, bins)` — count occurrences of each integer value; out-of-range values are dropped
- `where(cond, a, b)` — elementwise select

## Randomness

Random at run time, different every run; set `VECTOR_SEED=<n>` to reproduce a run exactly — the same seed gives the same values on every backend. Initializers (`randn`, `glorot_uniform`, ...) stay fixed at compile time so programs are testable.

- `uniform(dims...)` — uniform values in [0, 1]
- `normal(dims...)` — standard normal values (Box-Muller over uniform)
- `permutation(n)` — a random permutation of the indices 0..n; shuffle data with `take(x, permutation(n))`
- `random_windows(data, count, size)` — `count` random contiguous windows of `size` from axis 0; the data loading idiom for sequence models
- `dropout(x, rate)` — inverted dropout: keeps values with probability 1-rate and rescales; differentiable; becomes identity in `export`ed models
- `sample(logits)` — draw one index from a categorical distribution over a logits vector (Gumbel-max); batch with `vmap`; use `argmax` for greedy

## Neural networks

- `Linear(in_size, out_size)` — stdlib module: `w` (glorot uniform), `b` (zeros), `forward(self, x)`
- `LayerNorm(dim)` — stdlib module normalizing each row of a matrix: `gain`, `bias`, `forward(self, x)`; `token_norm(params, token)` is its per-row helper for direct `vmap` use
- `Embedding(count, dim)` — stdlib module: `w` (normal, std 0.02); calling it with an id vector gathers rows, differentiably
- `softmax_rows(m)` — softmax over each row of a matrix
- initializers: `glorot_uniform(fan_in, fan_out)`, `glorot_normal(fan_in, fan_out)`, `he_uniform(fan_in, fan_out)`, `he_normal(fan_in, fan_out)`, `lecun_uniform(fan_in, fan_out)`, `lecun_normal(fan_in, fan_out)`
- stdlib functions (rank-1, written in vector itself; batch with `vmap`): `relu(x)`, `sigmoid(x)`, `logsumexp(x)`, `var(x)`, `std(x)`, `norm(x)`, `layer_norm(x, gain, bias)`
- losses: `mse(pred, target)`; `cross_entropy(logits, target)` — from logits, target is a class index; batch with `vmap`
- optimizers hold their state in a record with the params at `.p`: `adam_init(model)` makes the state, `adam(st, grad, lr)` steps it; `adamw(st, grad, lr, decay)` decouples weight decay; `sgd_init(model)` + `sgd(st, grad, lr, momentum)`
- schedules are plain functions of the step: `cosine_decay(lr, step, total)`, `warmup(lr, step, steps)`
- clipping: `clip(x, lo, hi)`, `clip_by_norm(x, max_norm)`

## Text

Text enters as bytes or token ids — there are no strings in the graph.

- `load("data.txt")` — the file as a vector of byte values 0..255; byte-level models need no tokenizer (vocab 256)
- `tokenize("data.txt", "tokenizer.json")` — the file as a vector of token ids; any HuggingFace byte-level BPE tokenizer (gpt-2 family) works
- `detokenize(ids, "tokenizer.json")` — mark generated ids for printing as text: `print(detokenize(ids, "tokenizer.json"))`
- `text(x)` — mark a byte vector for printing as text: `print(text(x))`; `save(x, "out.txt")` writes it as a file
- tokenizers also build in vector itself: count pairs with `bincount`, merge with `where`, compact with `take(ids, argsort(dead))` — see the bpe example in [examples.md](examples.md)

## Files and network

`load("path")` and `save(value, "path")` dispatch on extension; `load` also accepts `https://` URLs (fetched once into `~/.vector/downloads`).

| Extension | Value shape |
| --------- | ----------- |
| `.npy` | one tensor (little-endian f32/f64, C order) |
| `.safetensors` | record or module instance — weights; PyTorch state_dicts load as records |
| `.csv` | record of f32 column vectors; text columns factorize to category codes |
| `.png` | image tensor `[h, w]` or `[h, w, c]`, f32 in 0..1 |
| `.wav` | record `{samples, rate}`, f32 in -1..1 |
| `.txt` | vector of byte values, f32 in 0..255 |
| `.gz` | gzipped idx (the mnist format) — tensor of byte values, f32 in 0..255 |

A `load` after a `save` of the same path in one program returns the saved value.

## Output and figures

- `print(x)` — print a tensor or record; inside a loop, logs one line per iteration
- `plot(y)` / `plot(x, y)` / `plot(x, y, "label")` — line series; `scatter(...)` for points
- `imshow(image)` — show an image tensor in the figure
- `title("t")`, `xlabel("x")`, `ylabel("y")`
- `savefig("fig.svg")` — write the figure; `show()` — write a temp file and open the viewer
- `play(clip)` — play a `{samples, rate}` record through the system player

## Deployment

- `export(model, "model.mlir", example_inputs...)` — write the forward pass as standalone StableHLO with weights baked in; example inputs fix the shapes
- `resize(image, height, width)` — bilinear, differentiable; `crop(image, top, left, height, width)`

## Command line

```
vector                        start the interactive repl
vector <file.vec>             compile and run a program
vector serve <m.mlir> [port]  serve an exported model over http
vector setup                  detect this machine and install the right backends
vector version                print version

--accelerate                  run on the machine's accelerator (gpu/tpu); cpu is the default
```

Environment: `VECTOR_BACKEND=<name>` pins a backend, `PJRT_PLUGIN_PATH` overrides the plugin, `XLA_FLAGS` passes through to XLA, `VECTOR_LOGS=1` shows the runtime logs vector hides, `VECTOR_SEED=<n>` pins the run-time randomness.
