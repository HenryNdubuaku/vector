Coding rules 
No comments. Names carry the meaning; if a line needs a comment, rename or split it.
Build only what the current milestone needs. No speculative abstraction, no config knobs, no "for later" hooks.
Functions by default. Introduce a class only when there is real state (Tracer, the jit cache).
One way to do each thing. No alternate code paths or compatibility shims.
Pure functions outside the tracer; no hidden global mutation.
Fewest lines that stay readable — prefer a comprehension or a dict dispatch over branching boilerplate, but never golf at the cost of clarity.
Flat layout, small files. Split only when a file stops being scannable in one screen.
Shape rules fail loud at trace time with the offending shapes in the message.
No dependency added without a concrete need; the pjrt crate (PJRT/StableHLO bindings) only.

Milestone 2 — make a forward pass expressible. The language can't express any actual ML yet. The gap list, roughly in dependency order: matmul (stablehlo.dot_general), unary math builtins (exp, log, tanh, sqrt, max for relu), axis-wise reductions (sum(x, axis), mean), and general broadcasting beyond scalar↔array ([2,3] + [3]). Broadcasting is a design decision, not just a feature: NumPy-style implicit alignment vs. stricter Dex-style explicitness — worth deciding consciously since it interacts with the static-rank direction you've chosen.

Milestone 3 — the differentiator: grad. This forces the one architectural change on the horizon: the tracer currently emits MLIR strings directly, but AD needs a structure you can transform — trace into a small op graph first, differentiate graph-to-graph, then emit. Per your own CLAUDE.md rule, don't build that IR speculatively — build it when grad lands, as its first consumer. vmap then reuses the same graph.

Milestone 4 — open the programs. Real use means data that isn't a source literal: program inputs, and eventually IO/PRNG/optimizer state. That's the effects question from your design notes — the biggest unresolved design problem, and it starts mattering exactly here. Related but smaller: today every run re-creates a PJRT client and recompiles; a REPL or repeated execution would motivate the jit cache.