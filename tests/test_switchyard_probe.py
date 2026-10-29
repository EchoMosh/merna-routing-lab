"""Real-library regressions plus separately identified fail-closed adapter tests."""

import asyncio
import copy
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import switchyard_probe as probe


def history(*, failed=False, tool="write_file", count=3):
    """Synthetic normalized tool traffic; no tools or commands are executed."""
    messages = [{"role": "user", "content": [{"type": "text", "text": "Synthetic code task."}]}]
    for i in range(count):
        messages.extend([
            {"role": "assistant", "content": [{
                "type": "tool_call", "id": f"call_{i}", "name": tool,
                "arguments": {"path": f"synthetic_{i}.py", "content": "x = 1"},
            }]},
            {"role": "user", "content": [{
                "type": "tool_result", "tool_call_id": f"call_{i}",
                "content": [{"type": "text", "text": (
                    "AssertionError: synthetic test failed" if failed else "File written successfully."
                )}], "is_error": failed,
            }]},
        ])
    return messages


def request(messages=None):
    return {"id": "synthetic", "messages": history() if messages is None else messages,
            "baseline_model": "synthetic/capable", "candidate_model": "synthetic/efficient"}


class RealLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            version = importlib.metadata.version("nemo-switchyard")
        except importlib.metadata.PackageNotFoundError:
            raise unittest.SkipTest("real-library cases require the pinned wheel")
        if version != probe.PINNED_VERSION:
            raise unittest.SkipTest("real-library cases require version " + probe.PINNED_VERSION)

    def test_successful_writes_use_real_upstream_dimensions(self):
        original = request()
        snapshot = copy.deepcopy(original)
        self.assertEqual(probe.select(original), {
            "selected_model": "synthetic/efficient",
            "decision_source": "switchyard.stage_router.dimensions", "engine_version": "0.3.0",
        })
        self.assertEqual(original, snapshot)

    def test_repeated_failures_use_real_upstream_override(self):
        result = probe.select(request(history(failed=True, tool="terminal")))
        self.assertEqual(result, {
            "selected_model": "synthetic/capable",
            "decision_source": "switchyard.stage_router.override", "engine_version": "0.3.0",
        })

    def test_openai_chat_history_normalizes_to_real_upstream(self):
        messages = [{"role": "user", "content": "Synthetic code task."}]
        for i in range(3):
            messages.extend([
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": f"c{i}", "type": "function", "function": {
                        "name": "write_file", "arguments": '{"path":"synthetic.py"}',
                    },
                }]},
                {"role": "tool", "tool_call_id": f"c{i}", "content": "File written successfully."},
            ])
        result = probe.select(request(messages))
        self.assertEqual(result["selected_model"], "synthetic/efficient")
        self.assertEqual(result["decision_source"], "switchyard.stage_router.dimensions")

    def test_ambiguous_read_history_retains_baseline(self):
        result = probe.select(request(history(tool="read_file", count=1)))
        self.assertEqual(result["selected_model"], "synthetic/capable")
        self.assertEqual(result["decision_source"], "abstain")
        self.assertNotIn("error", result)

    def test_cli_round_trip_uses_actual_library(self):
        script = Path(probe.__file__)
        process = subprocess.run(
            [sys.executable, str(script)], input=json.dumps(request()),
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(json.loads(process.stdout), probe.select(request()))
        self.assertEqual(process.stderr, "")


class AdapterSafetyTests(unittest.TestCase):
    """Mocks here test error handling; they are never reported as library routing."""

    def assert_abstains(self, result, error=None):
        self.assertEqual(result["selected_model"], "synthetic/capable")
        self.assertEqual(result["decision_source"], "abstain")
        if error is not None:
            self.assertEqual(result["error"], error)

    def setUp(self):
        # Validation cases remain runnable without the optional native wheel.
        self.version_patch = patch.object(probe.metadata, "version", return_value="0.3.0")
        self.version_patch.start()
        self.addCleanup(self.version_patch.stop)

    def test_broken_metadata_abstains_without_echoing_details(self):
        failures = [OSError("private sentinel"), UnicodeDecodeError("utf8", b"\xff", 0, 1, "private sentinel")]
        for failure in failures:
            with self.subTest(error=type(failure).__name__), patch.object(
                probe.metadata, "version", side_effect=failure
            ):
                result = probe.select(request())
                self.assert_abstains(result, "metadata_failure:" + type(failure).__name__)
                self.assertNotIn("private sentinel", str(result))

    def test_missing_dependency_retains_baseline(self):
        with patch.object(probe.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            result = probe.select(request())
        self.assert_abstains(result, "missing_dependency:nemo-switchyard")
        self.assertIsNone(result["engine_version"])

    def test_unpinned_version_retains_baseline(self):
        with patch.object(probe.metadata, "version", return_value="999.0.0"):
            result = probe.select(request())
        self.assert_abstains(result, "unsupported_engine_version")
        self.assertEqual(result["engine_version"], "999.0.0")

    def test_import_failure_retains_baseline(self):
        with patch.object(probe, "import_module", side_effect=ImportError("sensitive request text")):
            result = probe.select(request())
        self.assert_abstains(result, "upstream_failure:ImportError")
        self.assertNotIn("sensitive", json.dumps(result))

    def test_no_tool_history_abstains_before_import(self):
        with patch.object(probe, "import_module") as importer:
            result = probe.select(request([{"role": "user", "content": "Hello"}]))
        self.assert_abstains(result, "no_tool_history")
        importer.assert_not_called()

    def test_unknown_tools_abstain(self):
        self.assert_abstains(probe.select(request(history(tool="unknown_domain_tool"))), "unsupported_tool")

    def test_media_abstains(self):
        self.assert_abstains(probe.select(request([
            {"role": "user", "content": [{"type": "image", "source": {"url": "https://invalid"}}]},
        ])), "unsupported_content")

    def test_unmatched_results_abstain(self):
        messages = history()
        messages[2]["content"][0]["tool_call_id"] = "missing"
        self.assert_abstains(probe.select(request(messages)), "unmatched_tool_result")

    def test_incomplete_calls_abstain(self):
        self.assert_abstains(probe.select(request(history()[:-1])), "incomplete_tool_history")

    def test_bad_function_arguments_abstain(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": [{
            "id": "c1", "type": "function", "function": {"name": "write_file", "arguments": "{"},
        }]}]
        self.assert_abstains(probe.select(request(messages)), "invalid_tool_arguments")

    def test_running_event_loop_abstains(self):
        async def run():
            with patch.object(probe, "import_module"):
                return probe.select(request())
        self.assert_abstains(asyncio.run(run()), "running_event_loop:use_asyncio.to_thread")

    def test_same_model_ids_abstain(self):
        value = request()
        value["candidate_model"] = value["baseline_model"]
        self.assert_abstains(probe.select(value), "invalid_candidate_model")


if __name__ == "__main__":
    unittest.main()
