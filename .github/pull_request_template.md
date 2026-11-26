## Phase

PN.NN: <title> ([spec](../docs/m1/phases.md), [notes](../docs/m1/notes/))

## What changed

<!-- A few lines: what was built, and what to look at first. -->

## Gate

<!-- Paste the phase gate's summary: target/gates/<time>-phase/summary.md -->

## Definition of Done

- [ ] The scope matches the phase spec; any deviation is explained in the notes.
- [ ] The phase gate (G1) passed on the commit merged. G1c and G1p passed if the change reaches CUDA or Python.
- [ ] Tests cover the new behaviour. Anything that detects or supervises was broken once on purpose.
- [ ] Each example runs from a clean checkout as its README says.
- [ ] The notes, ADRs and manual test are written, and the indexes updated.
- [ ] Clean room: nothing copied or ported from OScope or any other codebase. The ideas taken are named in the notes.
- [ ] New dependencies' licences are checked, and each version is at least two weeks old.
- [ ] The measured size is reported against the estimate.
