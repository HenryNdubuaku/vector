# To-do

The core items are ordered: each unblocks the ones after it. The ecosystem tracks are independent of the core and of each other, contributors can pick any of them up.

## Staged loop execution
The standing architectural item — becomes necessary once training runs take minutes. Design premise: staging must not surrender the single-dispatch performance — stages exchange device buffer handles, never host data.
- [ ] Split programs at top-level loop boundaries: prefix / body / suffix executables, host-driven
- [ ] Loop-carried state stays device-resident and is donated (input/output aliasing, what `jax.jit(donate_argnums)` uses) so the step executable updates weights in place instead of copying
- [ ] Async dispatch: issue the next step while the host handles the previous one; block only where a value is observed (print, save, plot) — prefetching the next batch falls out of this
- [ ] Live per-epoch printing and a progress bar (rate, ETA); the buffered-print syntax goes live unchanged
- [ ] Checkpoint-every-k-epochs, early stopping (`while cond:` exists)
- [ ] Same treatment for the repl: session values become device buffer handles (kills the D2H/H2D round-trip per chunk)

## Mixed precision
The biggest raw lever left on accelerators (2–4x matmul throughput); whole-graph emission makes it a mechanical pass, not a rewrite.
- [ ] bf16 policy: params and reductions stay f32, dot/elementwise compute in bf16 — converts inserted systematically at emission
- [ ] Surface: `--bf16` flag or per-module policy; f32 stays the default everywhere
- [ ] Read bf16 safetensors (convert on load) — pretrained checkpoints are increasingly bf16
- [ ] Validate on the gpt example: bf16-vs-f32 loss curves agree, measure the speedup per backend

## Vision leftovers
Convolutions landed (stride-2 convs downsample, so nothing here blocks CNNs — examples/cnn.vec trains today); these round out the domain.
- [ ] max_pool / avg_pool → stablehlo.reduce_window; max_pool VJP = select_and_scatter
- [ ] pad(x, ...) builtin (sequence work needs it too)
- [ ] Batchnorm question: running stats are mutable state — decide the functional idiom (state-in-record, flax-style) before promising the layer
- [ ] A CNN on real images end to end (mnist is already loadable) — where it demos is your call

## Distributed ML
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
- [ ] Non-byte-level tokenizers (sentencepiece/metaspace, the llama family) — tokenize() dies loudly on them for now
- [ ] Indexing rank-3+ operands (`images[idx]` on `[n, h, w]`) — take handles rank 1–2 today, flatten first
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
