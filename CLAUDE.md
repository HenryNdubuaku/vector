Coding rules 
No comments. Names carry the meaning; if a line needs a comment, rename or split it.
Build only what the current milestone needs. No speculative abstraction, no config knobs, no "for later" hooks.
Functions by default. Introduce a class only when there is real state (Tracer, the jit cache).
One way to do each thing. No alternate code paths or compatibility shims.
Pure functions outside the tracer; no hidden global mutation.
Fewest lines that stay readable — prefer a comprehension or a dict dispatch over branching boilerplate, but never golf at the cost of clarity.
Flat layout, small files. Split only when a file stops being scannable in one screen.
Shape rules fail loud at trace time with the offending shapes in the message.
No dependency added without a concrete need; NumPy + the StableHLO/PJRT bindings only.