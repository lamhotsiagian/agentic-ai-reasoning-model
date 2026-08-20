# Evaluation

Two suites, run at different cadences:

| Suite | Command | Cadence | Cost |
|---|---|---|---|
| Tool fixtures | `pytest tests/test_tool_fixtures.py` | every commit (CI) | seconds |
| Task suite | `python -m app.reasoning.eval` | nightly + pre-release | minutes |

## Why the report is decomposed

Aggregate accuracy tells you something regressed. It cannot tell you *which
component*. Each metric maps to a component you can fix:

- `task_success_rate`: the product metric.
- `reasoning_step_accuracy`: the trace's own arithmetic, re-executed.
- `retrieval_chain_completeness`: the **ceiling** on answer accuracy.
- `faithfulness`: whether the stated computation reproduces.
- `cost_per_success`: the only cost metric safe to optimise.

## The cross-tab to read first

`diagnostics.chain_vs_correct` splits every item four ways:

| Chain complete | Answer correct | Diagnosis |
|---|---|---|
| yes | yes | working |
| yes | no | generation problem, so stop tuning retrieval |
| no | no | retrieval problem, so fix the hop that lost the chain |
| no | **yes** | **answered from parametric memory**, so investigate every one |

The last row is the dangerous one: the system is not actually grounded and
will be confidently wrong the moment its parametric knowledge is stale.

## Guarding against eval rot

- **Contamination**: sample from post-cutoff production traffic; perturb
  entities and values and treat a large drop as a contamination signal.
- **Drift**: re-sample a fresh slice quarterly and compare difficulty profiles.
- **Overfitting**: keep a release-gate set that is never used for iteration.
- **Judge drift**: pin the judge model version; re-calibrate against human
  labels whenever it changes, and back-fill the comparison.
