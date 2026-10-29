import copy
import json
import unittest

from evaluate import compare, shadow


def case(**changes):
    value = dict(id="synthetic-1", messages=[{"role": "user", "content": "private sentinel"}],
                 baseline_model="strong", candidate_model="small", family="coding",
                 candidate_capabilities_verified=True)
    value.update(changes)
    return value


def receipt(strategy, **changes):
    row = dict(task_id="one", run_id=strategy, suite_id="suite", family="documents",
               strategy=strategy, success=True, split="heldout", measurement="actual",
               validated_by="fixture-validator-v1", latency_ms=100, model_usd=1,
               router_usd=0, tool_usd=0)
    row.update(changes)
    return row


class ShadowTests(unittest.TestCase):
    def proposal(self, request):
        return dict(selected_model="small", decision_source="dimensions", engine_version="test")

    def test_shadow_never_applies_or_leaks_content(self):
        rows = [case(), case(id="mail", family="email"), case(id="pin", explicit_model=True),
                case(id="infra", infrastructure_failure=True), case(id="long", long_context=True),
                case(id="unverified", candidate_capabilities_verified=False)]
        before = copy.deepcopy(rows)
        report = shadow(rows, self.proposal)
        self.assertEqual(rows, before)
        self.assertEqual(report["decisions"][0]["guarded_proposal"], "small")
        self.assertTrue(all(r["guarded_proposal"] == "strong" for r in report["decisions"][1:]))
        self.assertTrue(all(not r["applied"] and r["retained_model"] == "strong" for r in report["decisions"]))
        self.assertNotIn("private sentinel", str(report))
        self.assertEqual(report["cost_savings"], "not_measured")

    def test_unknown_or_failed_selector_keeps_baseline(self):
        def fail(request):
            raise RuntimeError("secret sentinel")
        for selector in (fail, lambda r: {"selected_model": "unknown"},
                         lambda r: {"selected_model": "small", "error": "bad"}):
            report = shadow([case()], selector)
            self.assertEqual(report["decisions"][0]["guarded_proposal"], "strong")
            self.assertNotIn("secret sentinel", str(report))

    def test_duplicate_cases_rejected(self):
        with self.assertRaises(ValueError):
            shadow([case(), case()], self.proposal)

    def test_capability_flag_requires_a_boolean(self):
        for flag in ("false", 1, [True], {"yes": True}, None):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                shadow([case(candidate_capabilities_verified=flag)], self.proposal)
        missing = case()
        del missing["candidate_capabilities_verified"]
        self.assertEqual(shadow([missing], self.proposal)["decisions"][0]["guarded_proposal"], "strong")


class ComparisonTests(unittest.TestCase):
    def paired_rows(self, count=80):
        return [receipt(strategy, task_id=str(i), run_id=f"{i}-{strategy}",
                        model_usd=0) for i in range(count) for strategy in ("baseline", "candidate")]

    def test_equal_decimal_costs_do_not_count_as_savings(self):
        rows = self.paired_rows()
        rows[0].update(model_usd=.1, router_usd=.2)
        rows[1].update(model_usd=.3)
        report = compare(rows)
        self.assertEqual(report["total_usd"]["baseline"], report["total_usd"]["candidate"])
        self.assertIn("no_total_cost_reduction", report["blocking_reasons"])
        self.assertFalse(report["eligible_for_review"])
        json.dumps(report, allow_nan=False)

    def test_aggregate_overflow_rejected_and_mean_does_not_overflow(self):
        rows = self.paired_rows()
        for row in rows:
            row.update(latency_ms=1e307)
        report = compare(rows)
        self.assertEqual(report["mean_latency_ms"]["baseline"], 1e307)
        json.dumps(report, allow_nan=False)
        for row in rows:
            row.update(model_usd=1e307)
        with self.assertRaises(ValueError):
            compare(rows)

    def test_router_and_retry_costs_can_erase_savings(self):
        report = compare([receipt("baseline"), receipt("candidate", model_usd=.2, router_usd=.9)])
        self.assertAlmostEqual(report["total_usd"]["candidate"], 1.1)
        self.assertIn("no_total_cost_reduction", report["blocking_reasons"])
        self.assertIn("quality_regression_uncertainty_too_high", report["blocking_reasons"])
        self.assertFalse(report["production_approved"])

    def test_rescues_do_not_hide_regressions(self):
        rows = [receipt("baseline"), receipt("candidate", success=False, model_usd=.1),
                receipt("baseline", task_id="two", run_id="b2", success=False),
                receipt("candidate", task_id="two", run_id="c2", model_usd=.1)]
        report = compare(rows)
        self.assertEqual(report["regressions"], 1)
        self.assertEqual(report["rescues"], 1)
        self.assertFalse(report["eligible_for_review"])

    def test_bad_unpaired_estimated_training_and_nonfinite_receipts_rejected(self):
        bad = [dict(measurement="estimated"), dict(split="train"), dict(success="yes"),
               dict(router_usd=-1), dict(model_usd=float("nan")), dict(validated_by=""),
               dict(latency_ms=True), dict(suite_id="different"), dict(family="other")]
        for changes in bad:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                compare([receipt("baseline"), receipt("candidate", **changes)])
        for rows in ([receipt("baseline")], [], [receipt("baseline"), receipt("baseline")]):
            with self.assertRaises(ValueError):
                compare(rows)


if __name__ == "__main__":
    unittest.main()
