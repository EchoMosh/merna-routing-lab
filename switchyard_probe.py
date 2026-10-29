"""Offline, fail-closed adapter for the released NeMo Switchyard stage router.

The upstream Rust library makes each accepted routing decision. This adapter
only validates/normalizes history and maps capable/efficient to caller model IDs.
There is no model client, credential lookup, inference, or learned score here.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

PINNED_VERSION = "0.3.0"
PICKER = "capable_first"  # Experimental upstream; ambiguity stays on baseline.
CONFIDENCE_THRESHOLD = 0.3  # Probe setting, not calibrated for this model pair.

# A bounded subset of the pinned upstream built-in vocabulary, not a classifier.
# Unknown tools abstain rather than inferring semantics from arbitrary names.
# Source: crates/libsy/src/algorithms/util/tool_signals.rs at v0.3.0.
SUPPORTED_TOOLS = frozenset(
    {
        "edit", "multiedit", "notebookedit", "str_replace",
        "str_replace_based_edit_tool", "apply_patch", "text_editor", "patch",
        "write", "create_file", "new_file", "write_file",
        "read", "view", "read_file", "search_files",
        "todowrite", "todo_write", "todo", "update_plan",
        "bash", "shell_command", "shell", "local_shell_call", "terminal",
        "exec_command",
    }
)


class UnsupportedHistory(ValueError):
    """The request cannot be represented by this bounded adapter."""


def _text_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise UnsupportedHistory("unsupported_content")
    blocks = []
    for block in content:
        if (
            not isinstance(block, dict)
            or set(block) != {"type", "text"}
            or block.get("type") != "text"
            or not isinstance(block.get("text"), str)
        ):
            raise UnsupportedHistory("unsupported_content")
        blocks.append(dict(block))
    return blocks


def _normalize(messages: Any) -> list[dict[str, Any]]:
    """Accept normalized text/tool blocks and OpenAI Chat function histories.

    All calls must have exactly one matching result. The history is treated as
    data: tool arguments and output are never executed. Provider-native media,
    reasoning, continuations, and unknown message fields fail closed.
    """
    if not isinstance(messages, list) or not messages:
        raise UnsupportedHistory("missing_messages")
    normalized = []
    calls: dict[str, str] = {}
    results: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            raise UnsupportedHistory("invalid_message")
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise UnsupportedHistory("unsupported_role")
        allowed = {"role", "content"}
        if role == "tool":
            allowed |= {"tool_call_id", "is_error"}
        elif role == "assistant":
            allowed.add("tool_calls")
        if set(message) - allowed:
            raise UnsupportedHistory("unsupported_message_fields")

        if role == "tool":
            blocks = [{
                "type": "tool_result",
                "tool_call_id": message.get("tool_call_id"),
                "content": _text_blocks(message.get("content")),
                "is_error": message.get("is_error"),
            }]
        else:
            content = message.get("content")
            if content is None and role == "assistant" and message.get("tool_calls"):
                content = []
            if isinstance(content, str):
                blocks = _text_blocks(content)
            elif isinstance(content, list):
                blocks = copy.deepcopy(content)
            else:
                raise UnsupportedHistory("unsupported_content")
            if "tool_calls" in message:
                if not isinstance(message["tool_calls"], list):
                    raise UnsupportedHistory("invalid_tool_calls")
                for call in message["tool_calls"]:
                    if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
                        raise UnsupportedHistory("unsupported_tool_call")
                    function = call["function"]
                    if (
                        call["type"] != "function"
                        or not isinstance(function, dict)
                        or set(function) != {"name", "arguments"}
                    ):
                        raise UnsupportedHistory("unsupported_tool_call")
                    arguments = function["arguments"]
                    if not isinstance(arguments, str):
                        raise UnsupportedHistory("invalid_tool_arguments")
                    try:
                        arguments = json.loads(arguments)
                    except (ValueError, TypeError) as exc:
                        raise UnsupportedHistory("invalid_tool_arguments") from exc
                    blocks.append({
                        "type": "tool_call", "id": call["id"],
                        "name": function["name"], "arguments": arguments,
                    })

        for block in blocks:
            if not isinstance(block, dict):
                raise UnsupportedHistory("unsupported_content")
            kind = block.get("type")
            if kind == "text":
                _text_blocks([block])
            elif kind == "tool_call":
                if role != "assistant" or set(block) != {"type", "id", "name", "arguments"}:
                    raise UnsupportedHistory("invalid_tool_call")
                name, call_id = block["name"], block["id"]
                if not isinstance(name, str) or name.lower() not in SUPPORTED_TOOLS:
                    raise UnsupportedHistory("unsupported_tool")
                if not isinstance(call_id, str) or not call_id or call_id in calls:
                    raise UnsupportedHistory("invalid_tool_call_id")
                if not isinstance(block["arguments"], dict):
                    raise UnsupportedHistory("invalid_tool_arguments")
                calls[call_id] = name
            elif kind == "tool_result":
                if role not in {"user", "tool"} or set(block) - {
                    "type", "tool_call_id", "content", "is_error"
                }:
                    raise UnsupportedHistory("invalid_tool_result")
                call_id = block.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in calls or call_id in results:
                    raise UnsupportedHistory("unmatched_tool_result")
                if block.get("is_error") is not None and type(block["is_error"]) is not bool:
                    raise UnsupportedHistory("invalid_error_flag")
                block["content"] = _text_blocks(block.get("content"))
                results.add(call_id)
            else:
                raise UnsupportedHistory("unsupported_content")
        normalized.append({"role": role, "content": blocks})
    if not results:
        raise UnsupportedHistory("no_tool_history")
    if set(calls) != results:
        raise UnsupportedHistory("incomplete_tool_history")
    return normalized


async def _drive(libsy: Any, messages: list, baseline: str, candidate: str) -> tuple[str, str]:
    algorithm = libsy.algorithms.stage_router(
        picker=PICKER, confidence_threshold=CONFIDENCE_THRESHOLD, classifier=None,
    )
    stream = algorithm.run_stream(
        {"model": "offline-switchyard-probe", "messages": messages},
        {"capable": [baseline], "efficient": [candidate], "any": [baseline, candidate]},
    )
    async for step in stream:
        if isinstance(step, libsy.Step.CallModel):
            # Never supply a fabricated completion or serve a model call.
            step.call.fail(RuntimeError("offline_probe_forbids_model_calls"))
            raise RuntimeError("unexpected_model_call")
        if not isinstance(step, libsy.Step.Done):
            raise RuntimeError("unexpected_upstream_step")
        outcome = step.outcome
        models = outcome.selected_model_ids
        if not models or models[0] not in {baseline, candidate}:
            raise RuntimeError("invalid_upstream_selection")
        evidence = outcome.metadata.evidence if outcome.metadata is not None else None
        if not isinstance(evidence, dict) or evidence.get("source") not in {
            "dimensions", "override", "capable_hold", "fall_open",
        }:
            raise RuntimeError("unsupported_upstream_evidence")
        source = evidence["source"]
        if source == "fall_open":
            return baseline, "abstain"
        return models[0], "switchyard.stage_router." + source
    raise RuntimeError("missing_upstream_outcome")


def select(request: dict) -> dict:
    """Return a real offline upstream selection, or retain baseline and abstain.

    ``baseline_model`` is the capable tier; ``candidate_model`` is efficient.
    The caller establishes that relationship; the library does not inspect model
    availability, prices, capabilities, or quality. Use from a synchronous thread.
    """
    baseline = request.get("baseline_model") if isinstance(request, dict) else None
    result = {"selected_model": baseline, "decision_source": "abstain", "engine_version": None}
    try:
        result["engine_version"] = metadata.version("nemo-switchyard")
    except metadata.PackageNotFoundError:
        return {**result, "error": "missing_dependency:nemo-switchyard"}
    except Exception as exc:
        return {**result, "error": "metadata_failure:" + type(exc).__name__}
    if result["engine_version"] != PINNED_VERSION:
        return {**result, "error": "unsupported_engine_version"}
    if not isinstance(request, dict) or "id" not in request:
        return {**result, "error": "invalid_request"}
    candidate = request.get("candidate_model")
    if not isinstance(baseline, str) or not baseline.strip():
        return {**result, "error": "invalid_baseline_model"}
    if not isinstance(candidate, str) or not candidate.strip() or candidate == baseline:
        return {**result, "error": "invalid_candidate_model"}
    try:
        messages = _normalize(request.get("messages"))
        libsy = import_module("switchyard.libsy")
        # asyncio.run cannot nest; abstain explicitly without leaking a coroutine.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return {**result, "error": "running_event_loop:use_asyncio.to_thread"}
        selected, source = asyncio.run(_drive(libsy, messages, baseline, candidate))
        return {**result, "selected_model": selected, "decision_source": source}
    except UnsupportedHistory as exc:
        return {**result, "error": str(exc)}
    except Exception as exc:
        # Do not echo upstream errors which might include request content.
        return {**result, "error": "upstream_failure:" + type(exc).__name__}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="JSON request file, or - for stdin")
    args = parser.parse_args()
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
        request = json.loads(raw)
    except (OSError, ValueError):
        print(json.dumps({"selected_model": None, "decision_source": "abstain",
                          "engine_version": None, "error": "invalid_json_input"}))
        return 2
    print(json.dumps(select(request), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
