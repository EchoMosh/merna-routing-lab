"""Offline shadow decisions and paired whole-task result comparisons.

Shadow replay never calls completion models, changes Hermes, or estimates savings.
Comparison accepts independently validated, actually measured task receipts only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path


def load_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def shadow(cases, selector):
    """Return content-free receipts; caller owns persisting them privately."""
    rows, seen = [], set()
    for case in cases:
        cid = case["id"]
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError("case IDs must be nonempty and unique")
        seen.add(cid)
        baseline, candidate = case["baseline_model"], case["candidate_model"]
        if not all(isinstance(m, str) and m for m in (baseline, candidate)):
            raise ValueError("model IDs must be nonempty strings")
        if baseline == candidate:
            raise ValueError("baseline and candidate must differ")
        for flag in ("explicit_model", "infrastructure_failure", "long_context",
                     "candidate_capabilities_verified"):
            if flag in case and type(case[flag]) is not bool:
                raise ValueError(f"{flag} must be a boolean")
        # Do not supply private task metadata to an imported selector.
        request = {k: case[k] for k in ("id", "messages", "baseline_model", "candidate_model")}
        error = None
        proposed = {}
        try:
            proposed = selector(request)
            if not isinstance(proposed, dict):
                raise ValueError("invalid selector result")
            selected = proposed.get("selected_model")
            if selected not in (baseline, candidate):
                raise ValueError("unknown selection")
            if proposed.get("error"):
                raise ValueError("selector reported failure")
        except Exception as exc:
            # Never persist exception strings: upstream errors can contain prompts/keys.
            error = type(exc).__name__
            selected = baseline
            version = proposed.get("engine_version") if isinstance(proposed, dict) else None
            proposed = {"decision_source": "abstain", "engine_version": version or "unavailable"}

        reason = "upstream_proposal"
        effective = selected
        if case.get("explicit_model"):
            effective, reason = baseline, "explicit_model_pin"
        elif case.get("infrastructure_failure"):
            effective, reason = baseline, "repair_dependency_before_model_escalation"
        elif case.get("family") != "coding":
            effective, reason = baseline, "domain_not_validated"
        elif case.get("long_context", False):
            effective, reason = baseline, "cache_switch_cost_not_measured"
        elif not case.get("candidate_capabilities_verified", False):
            effective, reason = baseline, "candidate_capabilities_unverified"
        elif error:
            reason = "selector_abstained"
        if error:
            effective = baseline

        def bounded_string(value, fallback):
            return value if isinstance(value, str) and len(value) <= 120 else fallback

        rows.append({
            "case_id": cid, "input_sha256": digest(request), "mode": "offline_shadow",
            "engine_version": bounded_string(proposed.get("engine_version"), "unknown"),
            "raw_proposal": selected, "guarded_proposal": effective,
            "retained_model": baseline, "applied": False,
            "decision_source": bounded_string(proposed.get("decision_source"), "unknown"),
            "guard_reason": reason, "error_type": error,
        })
    return {
        "mode": "offline_shadow", "cases": len(rows), "live_model_changes": 0,
        "model_completion_calls": 0, "task_success": "not_measured",
        "cost_savings": "not_measured", "decisions": rows,
    }


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def json_number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("aggregate measurement exceeds finite JSON numeric range")
    return result


def wilson_upper(failures, count):
    """Upper end of a two-sided 95% Wilson binomial interval."""
    if count == 0:
        return 1.0
    z = 1.959963984540054
    p = failures / count
    return (p + z*z/(2*count) + z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))) / (1+z*z/count)


def compare(receipts, max_regression=0.05):
    """Compare paired complete runs, never splice hypothetical per-step outcomes."""
    if not 0 <= max_regression <= 1:
        raise ValueError("max_regression must be between zero and one")
    by_case, run_ids, suites = {}, set(), set()
    for row in receipts:
        required = {"task_id", "run_id", "suite_id", "family", "strategy", "success",
                    "split", "measurement", "validated_by", "latency_ms", "model_usd",
                    "router_usd", "tool_usd"}
        if not required <= row.keys():
            raise ValueError("incomplete receipt")
        for key in ("task_id", "run_id", "suite_id", "family", "validated_by"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        if row["split"] != "heldout" or row["measurement"] != "actual":
            raise ValueError("only actual held-out measurements are comparable")
        if type(row["success"]) is not bool:
            raise ValueError("success must be a validated boolean")
        if row["strategy"] not in ("baseline", "candidate"):
            raise ValueError("unknown strategy")
        if row["run_id"] in run_ids:
            raise ValueError("duplicate run ID")
        run_ids.add(row["run_id"])
        suites.add(row["suite_id"])
        key = row["task_id"]
        pair = by_case.setdefault(key, {})
        if row["strategy"] in pair:
            raise ValueError("duplicate task/strategy; aggregate repeats explicitly")
        pair[row["strategy"]] = dict(row)
        pair[row["strategy"]]["total_usd"] = sum(nonnegative(row[k], k) for k in
                                                      ("model_usd", "router_usd", "tool_usd"))
        pair[row["strategy"]]["latency_ms"] = nonnegative(row["latency_ms"], "latency_ms")
    if not by_case or len(suites) != 1:
        raise ValueError("one nonempty evaluation suite is required")
    counts = Counter()
    cost = {"baseline": Decimal(0), "candidate": Decimal(0)}
    latency = {"baseline": Decimal(0), "candidate": Decimal(0)}
    families = Counter()
    for pair in by_case.values():
        if set(pair) != {"baseline", "candidate"}:
            raise ValueError("every task needs both complete runs")
        a, b = pair["baseline"], pair["candidate"]
        if a["family"] != b["family"]:
            raise ValueError("paired task families disagree")
        families[a["family"]] += 1
        counts["baseline_successes"] += a["success"]
        counts["candidate_successes"] += b["success"]
        counts["regressions"] += a["success"] and not b["success"]
        counts["rescues"] += not a["success"] and b["success"]
        for strategy in cost:
            cost[strategy] += pair[strategy]["total_usd"]
            latency[strategy] += pair[strategy]["latency_ms"]
    n = len(by_case)
    upper = wilson_upper(counts["regressions"], counts["baseline_successes"])
    reasons = []
    if upper > max_regression:
        reasons.append("quality_regression_uncertainty_too_high")
    if counts["regressions"]:
        reasons.append("observed_task_regressions")
    if cost["candidate"] >= cost["baseline"]:
        reasons.append("no_total_cost_reduction")
    return {
        "mode": "paired_task_comparison", "suite_id": next(iter(suites)), "paired_tasks": n,
        "families": dict(families), **{k: counts[k] for k in
            ("baseline_successes", "candidate_successes", "regressions", "rescues")},
        "total_usd": {k: json_number(v) for k, v in cost.items()},
        "mean_latency_ms": {k: json_number(v/n) for k, v in latency.items()},
        "regression_upper_95": upper, "max_regression": max_regression,
        "eligible_for_review": not reasons, "blocking_reasons": reasons,
        "production_approved": False,
        "limitations": ["Receipt validator provenance must be independently checked.",
                        "Binomial interval assumes independent tasks; family coverage matters.",
                        "This report never activates routing; latency and live integration need review."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("shadow", "compare"):
        sub = subs.add_parser(name)
        sub.add_argument("input", type=Path)
        sub.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_rows(args.input)
    if args.command == "shadow":
        try:
            from switchyard_probe import select
        except ImportError:
            def select(request):
                raise RuntimeError("selector unavailable")
        report = shadow(rows, select)
    else:
        report = compare(rows)
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
