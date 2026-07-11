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

Milestone 4 — open the programs. Real use means data that isn't a source literal: program inputs, and eventually IO/PRNG/optimizer state. That's the effects question from your design notes — the biggest unresolved design problem, and it starts mattering exactly here. Related but smaller: today every run re-creates a PJRT client and recompiles; a REPL or repeated execution would motivate the jit cache.