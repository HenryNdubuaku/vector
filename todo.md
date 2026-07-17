# To-do

The core items are ordered: each unblocks the ones after it. The ecosystem tracks are independent of the core and of each other, contributors can pick any of them up.

## 1. Text and tokenizers
Tokenization is data prep, not differentiable compute — it lives at the boundary with the other codecs, never in the graph. No strings in the language.
- [ ] `load("data.txt")` → byte tensor; byte-level models need no tokenizer at all (vocab 256)
- [ ] Read HuggingFace `tokenizer.json` (JSON parser already exists); BPE encode host-side in Rust
- [ ] `tokenize("data.txt", "tokenizer.json")` → id tensor; `detokenize(ids, "tokenizer.json")` for printing generated text
- [ ] Until then, document the zero-work path: pre-tokenize with any tool → save ids as `.npy` → `load()`

## 2. Small language gaps
Cheap language work that makes everything after it read naturally — worth landing before the GPT demo so the demo code shows it off.
- [ ] Indexing sugar `x[i]` / `x[a:b]` as syntax for take/slice — the single biggest recognisability win for python users
- [ ] `while cond:` with a runtime condition (stablehlo.while already supports it; today only fixed trip counts) — also what early stopping needs later
- [ ] Builtins a torch user trips on immediately: abs, pow, concat, stack (the Concat op already exists in the graph — just not exposed)

## 3. The payoff demo: a small GPT in vector
Byte-level Shakespeare in `example/` — the artifact that makes people install it, and the load test that flushes out remaining gaps.
- [ ] Attention, layernorm, Adam written in vector source (softmax/mean/var/records already expressible) — promote the good ones to stdlib
- [ ] Causal mask via `where` + iota comparisons
- [ ] Train, sample, and print generated text end to end

## 4. NN training stdlib
What a torch/jax user reaches for daily, written in vector source (records already express optimizer state); harvests what the GPT demo proves out, while the code is fresh.
- [ ] Optimizers: adam, adamw, sgd with momentum — state as a record, `state, params = adam(state, params, grads, lr)` shape
- [ ] Losses: cross_entropy (from logits, via logsumexp), mse
- [ ] Schedules: cosine decay, linear warmup — plain functions of the step
- [ ] `normal(dims...)` runtime sampling (Box-Muller over uniform — pure stdlib source); unlocks VAEs/diffusion later
- [ ] Gradient clipping: clip(x, lo, hi) and clip_by_norm (norm exists)

## 5. Staged loop execution
The standing architectural item — becomes necessary once training runs take minutes (which the GPT demo will cause). Design premise: staging must not surrender the single-dispatch performance — stages exchange device buffer handles, never host data.
- [ ] Split programs at top-level loop boundaries: prefix / body / suffix executables, host-driven
- [ ] Loop-carried state stays device-resident and is donated (input/output aliasing, what `jax.jit(donate_argnums)` uses) so the step executable updates weights in place instead of copying
- [ ] Async dispatch: issue the next step while the host handles the previous one; block only where a value is observed (print, save, plot) — prefetching the next batch falls out of this
- [ ] Live per-epoch printing and a progress bar (rate, ETA); the buffered-print syntax goes live unchanged
- [ ] Checkpoint-every-k-epochs, early stopping (uses `while cond:` from item 2)
- [ ] Same treatment for the repl: session values become device buffer handles (kills the D2H/H2D round-trip per chunk)

## 6. Mixed precision
The biggest raw lever left on accelerators (2–4x matmul throughput); whole-graph emission makes it a mechanical pass, not a rewrite.
- [ ] bf16 policy: params and reductions stay f32, dot/elementwise compute in bf16 — converts inserted systematically at emission
- [ ] Surface: `--bf16` flag or per-module policy; f32 stays the default everywhere
- [ ] Read bf16 safetensors (convert on load) — pretrained checkpoints are increasingly bf16
- [ ] Validate on the GPT demo: bf16-vs-f32 loss curves agree, measure the speedup per backend

## 7. Convolutions and vision
The one op-family gap that closes off a whole domain; everything else in vision is composition.
- [ ] conv(x, kernel, stride, padding) → stablehlo.convolution; VJP = transposed conv (both directions needed for training)
- [ ] max_pool / avg_pool → stablehlo.reduce_window; max_pool VJP = select_and_scatter
- [ ] pad(x, ...) builtin (convs and sequence work both need it)
- [ ] Batchnorm question: running stats are mutable state — decide the functional idiom (state-in-record, flax-style) before promising the layer
- [ ] Small CNN on real images end to end (mnist-scale; the PNG codec already loads data) — where it demos is your call

## 8. Distributed ML
Enter with GSPMD: annotate shardings, XLA partitions the graph and inserts collectives — vector's whole-program graph is the ideal input.
- [ ] Optional early de-risk spike: enumerate PJRT devices, run one replicated executable on the 2×RTX box
- [ ] Single-host data parallel first (multi-GPU, multi-chip TPU)
- [ ] Multi-host after

## Ecosystem: notebooks
ML users live in notebooks; the repl already does the hard part (per-chunk compile, persistent session, recoverable errors) — a Jupyter kernel is a protocol wrapper around it.
- [ ] Jupyter kernel speaking ZMQ: cell = repl chunk, stdout streams as it prints, errors stay per-cell
- [ ] Inline figures: plot/imshow return SVG as display_data in notebook context instead of needing savefig
- [ ] `vector kernel install` writes the kernelspec so jupyter/VS Code discover it
- [ ] A notebook version of the sin walkthrough as the first artifact

## Ecosystem: editor support
- [ ] TextMate grammar for .vec (one JSON file, the syntax is small) — powers a VS Code extension and GitHub highlighting (submit to github-linguist)
- [ ] tree-sitter grammar — neovim/emacs and better GitHub rendering
- [ ] VS Code extension: grammar + run-file command; published to the marketplace
- [ ] LSP server later: the parser/tracer already give fast, precise errors — diagnostics on save, completion from builtins + stdlib
- [ ] `vector fmt` — canonical formatting; small language, small formatter

## Ecosystem: packages
Registry-free to start, the way Go proved out: a package is a git repo.
- [ ] A package = git repo of .vec files + a small manifest (name, version); `vector add <git-url>` fetches into ~/.vector/pkgs
- [ ] `import` gains a search path: file-relative first, then installed packages
- [ ] Publishing = pushing a version tag; `vector add` resolves tags; lockfile for reproducible builds
- [ ] A central index only if the ecosystem outgrows git URLs

## Ecosystem: distribution
- [ ] Prebuilt release binaries (macos arm64, linux x86_64/arm64) via CI on tag — install.sh downloads instead of compiling; kills the Rust-toolchain requirement, install drops to seconds
- [ ] Official docker images: cpu, cuda (bundling the vendored CUDA libs — pairs with the Standing item), tpu; built and pushed by CI on release; doubles as the reproducible benchmark environment
- [ ] Homebrew formula once release binaries exist

## Ecosystem: contributor infrastructure
- [ ] CI on every PR: cargo test on linux + macos runners (the golden suite is the merge gate)
- [ ] CONTRIBUTING.md: build steps, how the golden/docs-coverage tests work, the style rules (no comments, error strings carry the meaning)
- [ ] Label starter-sized items here and in issues as good-first

## Standing / smaller
- [ ] Vendored CUDA runtime: `vector setup` downloads NVIDIA redist libs into `~/.vector/cuda` (XLA already searches that path) — kills the apt/LD_LIBRARY_PATH dance
- [ ] Numerics check mode: opt-in flag that dies loudly on NaN/inf in any output, naming the op — debugging parity with torch's anomaly mode
- [ ] cumsum is O(n²) matmul — re-lower as a log-depth scan before GPT-scale sequence lengths
- [ ] mmap large `.npy` loads instead of reading them
- [ ] `load("config.json")` → record (parser exists; ~50 lines)
- [ ] CSV parser byte-level fast path (pandas is still 4× ahead)
- [ ] Table and plot builtins to parity with the pandas/matplotlib basics (legends, subplots, column ops)
- [ ] AWS Neuron backend (same wheel-extraction recipe as tpu; needs Trainium to validate)
- [ ] rocm / oneapi validation (needs hardware)
- [ ] PNG compression on write (files are valid but stored uncompressed)
