# ADR-0001: Recording decisions, and the delivery process

- Status: Accepted
- Date: 2026-11-26
- Phase: P1.01

## Context

- **Tetera is built in phases** by an agent, and reviewed by its owner one pull request at a time. A decision made in one phase must still be readable, with its reasons, many phases later.
- **OScope, the owner's previous build, showed what works** (its retrospectives): a spec before the code for the hard phases, an estimate in parts committed before the code, notes for every phase, and checklists for trying each phase by hand.
- **It also showed what doesn't:** line budgets that no phase fitted, and milestones of 50 to 65 phases.

## Decision

1. **Significant decisions are ADRs** in `docs/adr/`, from the template, numbered across the project, listed in the index. A phase's ADRs are *Accepted* when its pull request is approved. A later ADR may supersede one; neither is deleted.
2. **Phases follow plan §7:**
   - one branch and one pull request a phase: `m1/p1.07-short-name`, titled `P1.07: <title>`;
   - the owner approves every pull request, and merges it. Nothing is pushed to `main`, and no tag is made, without the owner's word;
   - before a phase's code, its second estimate goes into the plan's change log in a commit of its own, in five parts: code, tests, examples, docs-as-code and glue. Over 2,000 lines, the owner decides whether to split it;
   - spec-first phases open their notes with the behaviour and its literature before any code.
3. **Every pull request holds:** the code and its tests; examples with a README and a smoke mode; the phase's notes; its ADRs; its manual test; its status in the notes' index; the gate's summary in its description.
4. **Manual tests:** the agent runs each phase's and records the result in the notes. The owner runs each checkpoint's demo.

## Consequences

- **Gains:** decisions keep their reasons; a phase's size is decided before it starts, not discovered after; every phase can be tried by hand from its notes.
- **Costs:** each phase writes notes and an estimate commit, and some write ADRs, on top of the code.
- **To watch:** estimates against measured sizes, phase by phase, in the retrospective.

## Alternatives considered

- **Decisions in pull request descriptions only:** they scatter, and can't be superseded cleanly.
- **One design document, kept up to date:** its history and reasons blur as it is edited.
- **OScope's line budgets:** they measured phases against a size the phases never had.
