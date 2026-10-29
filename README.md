# Merna Routing Lab

An offline lab for evaluating model routing in tool-using agents. It currently
wraps NVIDIA NeMo Switchyard 0.3.0 and compares independently verified whole-task
outcomes and costs.

**Experimental:** this is a routing integration and evaluation harness. It is
not a production router, a trained model, or evidence of cost savings. The
included scenarios are synthetic; no customer data or credentials are needed.

## What it does

- Runs the actual pinned Switchyard library against recorded tool histories.
- Records proposed routing choices without sending requests to completion models.
- Retains the baseline when tools, capabilities or task domains are unsupported.
- Compares paired complete-task receipts, counting router/tool costs and retries.
- Reports regressions separately from rescues, with uncertainty for small samples.

The stage adapter uses upstream coding-oriented heuristics. Its confidence is
not a calibrated probability that a model will complete a task. The experimental
`capable_first` picker and `0.3` threshold are probe settings, not recommendations.

## Quick start

Python 3.10+ and a platform with a compatible Switchyard wheel are required.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python evaluate.py shadow fixtures/scenarios.jsonl --output runs/shadow.json
```

The binary-only installation avoids an accidental Rust build. Run tests in the
environment above: five upstream integration tests skip when the pinned library
is absent; a run with skips is not integration acceptance.

The nine included scenarios cover file progress, passing/failing tests, unknown
browser/email tasks, a model pin, long context and unverified tool capabilities.
They exercise code paths, not model intelligence. The model IDs are inert labels.

## Direct selector

```python
from switchyard_probe import select

result = select({
    "id": "example",
    "messages": recorded_tool_history,
    "baseline_model": "your-baseline-model",
    "candidate_model": "your-candidate-model",
})
```

See [the API and executed evidence](docs/SWITCHYARD-INTEGRATION.md) for supported
history shapes, actual library behavior, and limitations. Unknown inputs and
dependency failures abstain and retain the baseline. No model availability,
price or capability is inferred from its name.

## Compare completed jobs

Supply one JSONL receipt per **actually executed complete run**. Every task needs
a baseline and a candidate run; candidate receipts must come from routed runs,
not hypothetical combinations of outcomes from fixed-model runs.

| Field | Meaning |
| --- | --- |
| `task_id`, `run_id`, `suite_id`, `family` | Evaluation identifiers and task family |
| `strategy` | `baseline` or `candidate` |
| `split`, `measurement` | `heldout` and `actual` |
| `success`, `validated_by` | Boolean checked outcome and validator/version |
| `model_usd`, `router_usd`, `tool_usd` | Measured totals, including retries |
| `latency_ms` | Whole-run elapsed time |

```sh
.venv/bin/python evaluate.py compare private/receipts.jsonl --output runs/comparison.json
```

No actual model-run receipts are included. The tests invent numeric receipts to
check accounting; they are not benchmark results.

`eligible_for_review` requires zero observed regressions, lower total cost, and
a 95% Wilson upper bound on regressions among baseline successes no greater than
5%. It never approves production routing. Independence, family coverage, outcome
validation, latency and integration still need review. Adding more copies of the
same task does not establish reliability.

## Verification

26 tests passed with zero skips on the initial supported environment, including
five real-library checks. See [the synthetic replay](docs/OFFLINE-REPLAY.json).
Model quality and savings remain unmeasured. No live agent or provider settings
are changed by this repository.

Keep customer prompts, credentials, actual receipts and environments out of git.
`private/`, `runs/`, `.venv/` and `.env*` are ignored. The ignore rules are not a
substitute for checking files before committing them.

## License and attribution

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). This project uses the
[NVIDIA NeMo Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) package and is
an independent project, not an NVIDIA product. We credit the upstream routing
implementation; the wrapper and evaluation checks are the contribution here.
