# KeibaAI

- Python package in `src/keibaai`, tests in `tests/`. Run `pytest` after changes.

## Model roles
- Main session implements the change.
- Use the `scout` agent for file/code search; at most 3 in parallel. Ask for
  concise `file:line` summaries, not full files.
- Consult the advisor (Opus), when available, only when: finalizing a multi-file
  plan, the same test or error fails twice, or before an important commit.
  Not for routine edits.
- Run the `critic` agent before declaring done on multi-file changes or any change
  to features, model training, or backtests.
