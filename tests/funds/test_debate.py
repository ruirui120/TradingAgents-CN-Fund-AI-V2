import json
import unittest

from tradingagents.funds.debate import (
    ANALYST_ROLES,
    DeepSeekAgentBackend,
    ForecastDebateOrchestrator,
    _evidence_number_tokens,
    _number_tokens,
    _validated_claims,
    _validated_review,
)


EVIDENCE = [
    {"id": "FORECAST_H20", "type": "deterministic_historical_scenario", "data": {}},
    {"id": "RISK_METRICS", "type": "deterministic_calculation", "data": {}},
]


class FakeBackend:
    def __init__(
        self,
        invalid_role=None,
        unsafe_role=None,
        failing_role=None,
        invented_number_role=None,
        invalid_adjudication_once=False,
        invalid_adjudication_always=False,
        invalid_review_once=False,
        invalid_review_always=False,
        review_passed=True,
        extra_invented_number_role=None,
        invented_rebuttal_role=None,
    ):
        self.invalid_role = invalid_role
        self.unsafe_role = unsafe_role
        self.failing_role = failing_role
        self.invented_number_role = invented_number_role
        self.invalid_adjudication_once = invalid_adjudication_once
        self.invalid_adjudication_always = invalid_adjudication_always
        self.invalid_review_once = invalid_review_once
        self.invalid_review_always = invalid_review_always
        self.review_passed = review_passed
        self.extra_invented_number_role = extra_invented_number_role
        self.invented_rebuttal_role = invented_rebuttal_role
        self.calls = {}
        self.call_counts = {}

    def invoke(self, role, phase, payload):
        self.calls[(role, phase)] = payload
        self.call_counts[(role, phase)] = self.call_counts.get((role, phase), 0) + 1
        if role == self.failing_role and phase == "opinion":
            raise RuntimeError("simulated failure")
        if phase == "opinion":
            visible_id = payload["evidence_records"][0]["id"]
            evidence_id = "UNKNOWN" if role == self.invalid_role else visible_id
            statement = "建议立即卖出" if role == self.unsafe_role else f"{role} 有证据观点"
            if role == self.invented_number_role:
                statement = "上涨概率为百分之八十八（88%）"
            claims = [{"statement": statement, "evidence_ids": [evidence_id]}]
            if role == self.extra_invented_number_role:
                claims.append(
                    {
                        "statement": "额外声称上涨概率为 88%",
                        "evidence_ids": [evidence_id],
                    }
                )
            return {
                "stance": "neutral",
                "confidence": 0.5,
                "claims": claims,
                "risks": [],
                "limitations": ["历史情景不代表未来"],
            }
        if phase == "rebuttal":
            challenges = [
                {
                    "statement": f"{role} 对对方证据提出质疑",
                    "evidence_ids": ["FORECAST_H20"],
                }
            ]
            if role == self.invented_rebuttal_role:
                challenges.append(
                    {
                        "statement": "额外声称上涨概率为 88%",
                        "evidence_ids": ["FORECAST_H20"],
                    }
                )
            return {
                "challenges": challenges,
                "concessions": ["承认样本存在局限"],
            }
        if phase == "review":
            if self.invalid_review_always or (
                self.invalid_review_once
                and self.call_counts[(role, phase)] == 1
            ):
                return {
                    "passed": True,
                    "issues": [],
                    "accepted_evidence_ids": ["FORECAST_H20"],
                    "rejected_items": [],
                    "duplicate_evidence_groups": None,
                }
            return {
                "passed": self.review_passed,
                "issues": [],
                "accepted_evidence_ids": ["FORECAST_H20", "RISK_METRICS"],
                "rejected_items": [],
                "duplicate_evidence_groups": [],
            }
        if phase == "adjudication":
            if self.invalid_adjudication_always or (
                self.invalid_adjudication_once
                and self.call_counts[(role, phase)] == 1
            ):
                return {
                    "stance": "neutral",
                    "confidence": "low",
                    "summary": "H20 证据仍有分歧。",
                    "consensus": [],
                    "disagreements": [],
                    "invalidation_conditions": [],
                }
            return {
                "stance": "neutral",
                "confidence": "low",
                "summary": "现有证据存在分歧，应保留不确定性。",
                "consensus": ["历史情景不能保证未来"],
                "disagreements": ["趋势能否延续"],
                "invalidation_conditions": ["数据口径或基金策略发生变化"],
            }
        raise AssertionError((role, phase))


class ForecastDebateTests(unittest.TestCase):
    def test_backend_promotes_repair_instruction_to_trusted_system_message(self):
        class CapturingLlm:
            def invoke(self, messages):
                self.messages = messages
                return type(
                    "Response",
                    (),
                    {
                        "content": (
                            '{"passed":false,"issues":[],'
                            '"accepted_evidence_ids":[],"rejected_items":[],'
                            '"duplicate_evidence_groups":[]}'
                        )
                    },
                )()

        backend = DeepSeekAgentBackend.__new__(DeepSeekAgentBackend)
        backend.llm = CapturingLlm()

        backend.invoke(
            "reviewer",
            "review",
            {
                "evidence_records": EVIDENCE,
                "correction_instruction": "只修复 JSON 结构。",
            },
        )

        system_message = backend.llm.messages[0][1]
        human_payload = json.loads(backend.llm.messages[1][1])
        self.assertIn("只修复 JSON 结构。", system_message)
        self.assertNotIn("correction_instruction", human_payload)

    def test_complete_debate_never_changes_python_probabilities(self):
        result = ForecastDebateOrchestrator(FakeBackend()).run(EVIDENCE)

        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.opinions), len(ANALYST_ROLES))
        self.assertEqual(len(result.rebuttals), 2)
        self.assertTrue(result.review.passed)
        self.assertFalse(result.adjudication.probability_adjustment_applied)

    def test_claim_with_unknown_evidence_is_rejected_and_review_is_degraded(self):
        result = ForecastDebateOrchestrator(
            FakeBackend(invalid_role="tail_risk")
        ).run(EVIDENCE)

        self.assertEqual(result.status, "degraded")
        self.assertFalse(result.review.passed)
        self.assertTrue(any("无有效证据编号" in item for item in result.review.issues))

    def test_unsafe_agent_output_is_excluded(self):
        result = ForecastDebateOrchestrator(FakeBackend(unsafe_role="bear")).run(EVIDENCE)

        self.assertEqual(result.status, "degraded")
        self.assertFalse(any(item.role == "bear" for item in result.opinions))
        self.assertTrue(any("bear Agent" in item for item in result.failures))

    def test_agent_api_failure_degrades_instead_of_losing_base_forecast(self):
        result = ForecastDebateOrchestrator(
            FakeBackend(failing_role="quantitative")
        ).run(EVIDENCE)

        self.assertEqual(result.status, "degraded")
        self.assertTrue(any("quantitative Agent 调用失败" in item for item in result.failures))

    def test_number_missing_from_referenced_evidence_is_rejected(self):
        result = ForecastDebateOrchestrator(
            FakeBackend(invented_number_role="bull")
        ).run(EVIDENCE)

        self.assertEqual(result.status, "degraded")
        self.assertFalse(result.review.passed)
        self.assertTrue(any("不存在的数字" in item for item in result.review.issues))
        self.assertTrue(any("88" in item for item in result.review.issues))

    def test_data_holdings_and_tail_risk_agents_are_blind_to_base_forecast(self):
        backend = FakeBackend()
        ForecastDebateOrchestrator(backend).run(EVIDENCE)

        for role in ("data_auditor", "holdings_industry", "tail_risk"):
            visible_ids = {
                item["id"]
                for item in backend.calls[(role, "opinion")]["evidence_records"]
            }
            self.assertNotIn("FORECAST_H20", visible_ids)
        self.assertIn(
            "FORECAST_H20",
            {
                item["id"]
                for item in backend.calls[("quantitative", "opinion")][
                    "evidence_records"
                ]
            },
        )

    def test_number_tokens_do_not_extract_identifier_suffixes_or_negative_dates(self):
        self.assertEqual(_number_tokens("FORECAST_H20 return_p10_pct"), set())
        self.assertEqual(
            _number_tokens("截至 2026-06-30，情景收益为 -12.34%。"),
            {"2026", "6", "30", "-12.34"},
        )
        self.assertEqual(_number_tokens("共 3,860 个观测日"), {"3860"})

    def test_evidence_numbers_allow_only_exact_or_report_precision_values(self):
        record = {
            "id": "FORECAST_H20",
            "type": "deterministic_historical_scenario",
            "disclosure_period": "2026Q2",
            "data": {"return_p10_pct": -12.3456},
        }
        evidence_numbers = {"FORECAST_H20": _evidence_number_tokens(record)}

        claims, issues = _validated_claims(
            [
                {
                    "statement": "披露期为 2026 年第 2 季度，P10 为 -12.35%。",
                    "evidence_ids": ["FORECAST_H20"],
                }
            ],
            evidence_numbers,
            "test",
        )

        self.assertEqual(len(claims), 1)
        self.assertEqual(issues, [])
        self.assertNotIn("20", evidence_numbers["FORECAST_H20"])
        self.assertIn("10", evidence_numbers["FORECAST_H20"])

    def test_claim_with_any_unknown_evidence_id_is_rejected(self):
        evidence_numbers = {
            "RISK_METRICS": _evidence_number_tokens(
                {"id": "RISK_METRICS", "data": {"observation_count": 3860}}
            )
        }

        claims, issues = _validated_claims(
            [
                {
                    "statement": "共有 3,860 个观测日。",
                    "evidence_ids": ["RISK_METRICS", "UNKNOWN"],
                }
            ],
            evidence_numbers,
            "test",
        )

        self.assertEqual(claims, [])
        self.assertTrue(any("未知或无有效证据编号" in item for item in issues))

    def test_invalid_adjudication_is_repaired_once_without_changing_probabilities(self):
        backend = FakeBackend(invalid_adjudication_once=True)

        result = ForecastDebateOrchestrator(backend).run(EVIDENCE)

        self.assertEqual(result.status, "complete")
        self.assertIsNotNone(result.adjudication)
        self.assertFalse(result.adjudication.probability_adjustment_applied)
        self.assertEqual(backend.call_counts[("judge", "adjudication")], 2)

    def test_invalid_adjudication_degrades_after_one_repair_attempt(self):
        backend = FakeBackend(invalid_adjudication_always=True)

        result = ForecastDebateOrchestrator(backend).run(EVIDENCE)

        self.assertEqual(result.status, "degraded")
        self.assertIsNone(result.adjudication)
        self.assertEqual(backend.call_counts[("judge", "adjudication")], 2)

    def test_judge_is_not_called_when_review_fails(self):
        backend = FakeBackend(review_passed=False)

        result = ForecastDebateOrchestrator(backend).run(EVIDENCE)

        self.assertEqual(result.status, "degraded")
        self.assertFalse(result.review.passed)
        self.assertIsNone(result.adjudication)
        self.assertEqual(backend.call_counts[("reviewer", "review")], 1)
        self.assertNotIn(("judge", "adjudication"), backend.call_counts)

    def test_invalid_review_is_repaired_once_before_adjudication(self):
        backend = FakeBackend(invalid_review_once=True)

        result = ForecastDebateOrchestrator(backend).run(EVIDENCE)

        self.assertEqual(result.status, "complete")
        self.assertTrue(result.review.passed)
        self.assertIsNotNone(result.adjudication)
        self.assertEqual(backend.call_counts[("reviewer", "review")], 2)
        self.assertIn(
            "correction_instruction",
            backend.calls[("reviewer", "review")],
        )

    def test_invalid_review_degrades_after_one_repair_attempt(self):
        backend = FakeBackend(invalid_review_always=True)

        result = ForecastDebateOrchestrator(backend).run(EVIDENCE)

        self.assertEqual(result.status, "degraded")
        self.assertFalse(result.review.passed)
        self.assertIsNone(result.adjudication)
        self.assertEqual(backend.call_counts[("reviewer", "review")], 2)
        self.assertNotIn(("judge", "adjudication"), backend.call_counts)

    def test_string_false_review_never_authorizes_adjudication(self):
        backend = FakeBackend(review_passed="false")

        result = ForecastDebateOrchestrator(backend).run(EVIDENCE)

        self.assertEqual(result.status, "degraded")
        self.assertFalse(result.review.passed)
        self.assertIsNone(result.adjudication)
        self.assertEqual(backend.call_counts[("reviewer", "review")], 2)
        self.assertNotIn(("judge", "adjudication"), backend.call_counts)

    def test_review_rejects_unknown_accepted_evidence_id(self):
        raw = {
            "passed": True,
            "issues": [],
            "accepted_evidence_ids": ["UNKNOWN"],
            "rejected_items": [],
            "duplicate_evidence_groups": [],
        }

        with self.assertRaises(ValueError):
            _validated_review(raw, {"FORECAST_H20"}, [])

    def test_review_rejects_invalid_duplicate_evidence_groups(self):
        base = {
            "passed": True,
            "issues": [],
            "accepted_evidence_ids": ["FORECAST_H20"],
            "rejected_items": [],
        }
        invalid_groups = (
            [["FORECAST_H20", "UNKNOWN"]],
            [[{"id": "FORECAST_H20"}]],
        )

        for groups in invalid_groups:
            with self.subTest(groups=groups):
                with self.assertRaises(ValueError):
                    _validated_review(
                        {**base, "duplicate_evidence_groups": groups},
                        {"FORECAST_H20"},
                        [],
                    )

    def test_partial_invalid_opinion_claim_is_excluded_without_blocking_review(self):
        backend = FakeBackend(extra_invented_number_role="quantitative")

        result = ForecastDebateOrchestrator(backend).run(EVIDENCE)

        self.assertEqual(result.status, "complete")
        self.assertTrue(result.review.passed)
        self.assertTrue(any("88" in item for item in result.review.issues))
        quantitative = next(
            item for item in result.opinions if item.role == "quantitative"
        )
        self.assertEqual(len(quantitative.claims), 1)

    def test_partial_invalid_rebuttal_is_excluded_without_blocking_review(self):
        backend = FakeBackend(invented_rebuttal_role="bull")

        result = ForecastDebateOrchestrator(backend).run(EVIDENCE)

        self.assertEqual(result.status, "complete")
        self.assertTrue(result.review.passed)
        self.assertTrue(any("88" in item for item in result.review.issues))
        bull = next(item for item in result.rebuttals if item.role == "bull")
        self.assertEqual(len(bull.challenges), 1)


if __name__ == "__main__":
    unittest.main()
