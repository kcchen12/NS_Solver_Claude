# Branch Experiment Report

Date: 2026-04-02

## Goal
Split work into three branches so stable progress can be preserved while keeping risky experiments isolated.

## Branches

### branch-a-vorticity-only
- Purpose: candidate for main after one more validation run.
- Includes:
  - Vorticity contour replacement in plot panel 1.
- Excludes:
  - Nonuniform-grid feature work.
  - Added fail-infrastructure/checkers.
  - IBM soft-stabilization scaffolding.

### branch-b-vorticity-failinfra
- Purpose: preserve diagnostics/safety infrastructure without nonuniform changes.
- Includes:
  - Vorticity plotting change.
  - Analysis fail-infrastructure (force-source selector, mixed snapshot checks).
  - Runtime safety controls used during stabilization/debugging.
- Excludes:
  - Nonuniform-grid feature files and config surface.

### branch-c-vorticity-fail-nonuniform
- Purpose: full experiment stack.
- Includes:
  - Vorticity plotting.
  - Fail-infrastructure additions.
  - Nonuniform-grid implementation and related plumbing/tests/docs.
  - IBM soft-fix scaffolding used for nonuniform+IBM stabilization experiments.

## What likely affected drag behavior
Primary likely cause:
- Solver IBM handling/scaffolding path in the experimental stack changed force behavior and interacted with long-run pressure drift.

Secondary contributor:
- Analysis force-source behavior changes made drift visible differently depending on pressure vs metadata source.

## Practical recommendation
- Validate on branch-a-vorticity-only first.
- If drag behavior is acceptable there, merge/push that baseline.
- Keep branch-b and branch-c as reference/experiment branches.

## Repro/validation checklist
1. Clear old outputs before each run.
2. Run a baseline case on branch A and inspect drag/lift history.
3. Compare same case on branch B and C.
4. Record which branch first introduces undesired drag trend.
