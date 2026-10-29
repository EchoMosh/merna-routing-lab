# Offline Switchyard spike

Verified 2026-10-29. This probe executes the released NVIDIA NeMo Switchyard
native Python library on synthetic tool histories. It does not call an answer
model, judge model, tool, gateway, or service, and requires no credentials.
There is no live-routing activation or measured model-quality improvement.

## Release and installation evidence

- Distribution: `nemo-switchyard==0.3.0`; import surface: `switchyard.libsy`.
- [Upstream v0.3.0 release](https://github.com/NVIDIA-NeMo/Switchyard/releases/tag/v0.3.0),
  published 2026-09-22. The release installation instructions supersede stale
  source-build notices still present in the tagged README and stage-router docs.
  The annotated release tag resolves to source commit
  `336196f6fbfc97ddc71c1700f6092e564e9f23c2`.
- [Tagged Python package metadata](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.3.0/pyproject.toml):
  Python >=3.10; zero runtime dependencies; Rust extension built with maturin.
- Installed an existing 8,251,239-byte
  `nemo_switchyard-0.3.0-cp310-abi3-macosx_11_0_arm64.whl` on
  macOS 15.7.4 / arm64 / Python 3.15.0. Import and actual routing succeeded.
  No compiler, GPU, model assets, or additional Python packages were installed.
- [PyPI release metadata](https://pypi.org/pypi/nemo-switchyard/0.3.0/json)
  advertises wheel SHA-256
  `cdc64503e427095c73e7840f0de093f0ee35fc3e306825e522050cd6e6825039`.
  This is provenance from PyPI metadata, not a separately computed local hash.
  The loaded extension is `switchyard_rust/_switchyard_rust.abi3.so`.

Reproduce in the isolated repository environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --only-binary=:all: -r requirements.txt
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p test_switchyard_probe.py -v
```

The binary-only flag prevents an accidental source build on an unsupported
platform. `.venv` is local and uncommitted. No fallback to another library or
unreleased source exists. A missing/unusable library makes the adapter abstain.

## Current Python API

The [tagged Python bindings](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.3.0/switchyard_rust/libsy.py)
and [upstream binding tests](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.3.0/tests/test_libsy_minimal_bindings.py)
define the API used here:

```python
from switchyard.libsy import Step, algorithms

algorithm = algorithms.stage_router(
    picker="capable_first", confidence_threshold=0.3, classifier=None,
)
async for step in algorithm.run_stream(
    {"model": "offline-switchyard-probe", "messages": normalized_messages},
    {"capable": [baseline], "efficient": [candidate], "any": [baseline, candidate]},
):
    if isinstance(step, Step.Done):
        ordered_models = step.outcome.selected_model_ids
        evidence = step.outcome.metadata.evidence
    elif isinstance(step, Step.CallModel):
        # The adapter fails/abstains; it never serves this call.
        step.call.fail(RuntimeError("offline_probe_forbids_model_calls"))
```

The factory also exposes `recent_window`, handoff notes, per-tier system
prompts, `tool_semantics`, and an optional classifier. They are left at their
defaults here. Python exposes `run_stream`, not a synchronous `.run` method.
The Python factory does not expose the Rust/TOML `capable_hold_turns` option.
The complete [upstream embedding example](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.3.0/examples/libsy.py)
also describes serving model calls, which this probe deliberately never does.

The native request uses content blocks, including:

```json
[
  {"role":"assistant","content":[{"type":"tool_call","id":"c1","name":"write_file","arguments":{"path":"synthetic.py"}}]},
  {"role":"user","content":[{"type":"tool_result","tool_call_id":"c1","content":[{"type":"text","text":"File written successfully."}],"is_error":false}]}
]
```

These are data records; the probe never executes `write_file` or any command.
See the tagged [normalized protocol](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.3.0/crates/protocol/src/llm.rs).

## Adapter contract

```python
from switchyard_probe import select
result = select({
    "id": "synthetic-case",
    "messages": messages,
    "baseline_model": "synthetic/capable",
    "candidate_model": "synthetic/efficient",
})
```

Returns `selected_model`, `decision_source`, `engine_version`, and optionally
`error`. The installed distribution version is reported; when absent it is
`null`. The baseline model is returned unchanged on abstention/error. Callers
must provide the baseline/candidate tier relationship: Switchyard does not
verify arbitrary model IDs, capabilities, prices, account access, or endpoints.

- Actual decisions use `switchyard.stage_router.dimensions`,
  `switchyard.stage_router.override`, or `switchyard.stage_router.capable_hold`.
- Upstream `fall_open` becomes `abstain` with the baseline retained. This is
  adapter policy, not a learned classification. No confidence/quality score is
  returned by the contract.
- Supported inputs are normalized text/tool blocks and OpenAI Chat function
  calls/results. Chat string content and JSON function arguments are normalized
  to the upstream protocol without altering the caller's request.
- Supported tool names are the explicit bounded `SUPPORTED_TOOLS` set in
  `switchyard_probe.py`, copied from the tagged built-in vocabulary. It includes
  Hermes `write_file`, `patch`, `read_file`, `search_files`, `terminal` and
  Codex `apply_patch`, `exec_command`, `update_plan`. This allowlist gates input;
  it does not implement upstream error patterns, stage features, or scoring.
- Unknown tools, plain chat without tool results, missing or unmatched call
  IDs, duplicate results, pending calls, malformed arguments, native media,
  reasoning/continuation blocks, and unsupported message fields abstain.
  No custom tool semantics or model inference is guessed.
- Every call creates a fresh algorithm. Request `id` is an opaque fixture ID,
  not a session identifier. There is no cross-request affinity/recovery state.
- `select` is synchronous. In an existing asyncio loop use
  `await asyncio.to_thread(select, request)`; direct nested-loop use abstains.
- No credentials, client construction, inference, or network transport exist
  in the selection path. Unexpected `CallModel`, invalid upstream results,
  extension errors, and version mismatches all retain the baseline.
- Errors are short reason codes or exception class names, without echoing
  messages, tool output, or exception text into the result.

CLI: `.venv/bin/python switchyard_probe.py request.json` or pass one JSON
request through stdin with no argument. It emits one JSON object and exits 0
for a selection or abstention. Invalid JSON/file input exits 2 with an
abstention object; no baseline can be recovered from unreadable input.

## Observed synthetic results

These were executed through the installed native library, not mocked. The
model IDs are inert synthetic labels, not verified products or endpoints.

| History | Pinned upstream outcome | Adapter result |
|---|---|---|
| Three successful `write_file` call/result pairs | `['synthetic/efficient', 'synthetic/capable']`; evidence `source=dimensions`, `confidence=0.46211715726000974`, `threshold=0.3` | Candidate; `switchyard.stage_router.dimensions` |
| Three failed `terminal` pairs with explicit `is_error=true` and `AssertionError: synthetic test failed` | `['synthetic/capable', 'synthetic/efficient']`; evidence `source=override` | Baseline; `switchyard.stage_router.override` |
| One clean `read_file` pair | Capable first; evidence `source=fall_open` | Baseline; `abstain` |
| No tool history | Adapter rejects before invoking upstream | Baseline; `abstain`, `error=no_tool_history` |
| Unknown tool name | Adapter rejects before invoking upstream | Baseline; `abstain`, `error=unsupported_tool` |

For comparison, direct upstream `efficient_first` / threshold `0.5` selected
efficient with `source=fall_open` on successful writes **and on plain chat**;
the repeated-failure trace selected capable with `source=override`. These are
actual exploratory upstream results, not the shipped adapter configuration.
They explain why a generic efficient default is not treated as evidence that
the candidate can solve an unknown request.

The [stage-router implementation](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.3.0/crates/libsy/src/algorithms/util/stage.rs)
uses fixed weights and a tanh transform. The reported confidence above is a
heuristic magnitude, not a trained probability, learned model-quality score,
or measured success rate. Error overrides and tool semantics come from the
[upstream tool signal extractor](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.3.0/crates/libsy/src/algorithms/util/tool_signals.rs).

## Constraints and test evidence

`capable_first` is explicitly experimental upstream. The probe threshold
`0.3` permits clean production signals to choose efficient, and is not a
recommendation calibrated for these synthetic model labels. The tagged
[stage-router guide](https://github.com/NVIDIA-NeMo/Switchyard/blob/v0.3.0/docs/routing_algorithms/stage_router_routing.md)
recommends `efficient_first` / `0.5` as its benchmarked starting point; adopting
that policy would change unknown-request behavior. Measure real task quality
and model-pair outcomes separately before considering any deployment.

This lane tests API feasibility and deterministic routing only. It cannot
establish answer quality, financial savings, latency under real inference,
reliable switching between models, or production readiness. The root-owned
evaluation harness/fixtures are separate. No fixtures or live configuration
were changed by this lane.

Initial validation: **16 tests passed** with the command above. Five cases use
the real installed library (native success, native failure override, normalized
OpenAI Chat history, ambiguous history, and CLI). Eleven adapter safety cases
exercise abstention and error handling; their mocks are never represented as
actual upstream routing. Without the pinned library the five native cases
explicitly skip; run them in `.venv` to satisfy the integration gate.
