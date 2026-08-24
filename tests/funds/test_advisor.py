from datetime import date, datetime, timedelta, timezone
import unittest

from tradingagents.funds.advisor import ChinaFundAdvisor, render_markdown
from tradingagents.funds.models import (
    DebateReview,
    ForecastAdjudication,
    ForecastDebate,
    FundDataset,
    FundHolding,
    FundProfile,
    NavPoint,
    SourceRecord,
)


class FakeDataSource:
    def fetch(self, fund_code, as_of=None):
        return FundDataset(
            profile=FundProfile(
                code=fund_code,
                name="测试指数基金A",
                category="指数型",
                benchmark="沪深300指数收益率*95%+银行活期存款利率*5%",
            ),
            nav_history=[
                NavPoint(
                    nav_date=date(2024, 1, 1) + timedelta(days=index * 7),
                    unit_nav=1 + index * 0.002,
                )
                for index in range(60)
            ],
            holdings=[
                FundHolding(
                    asset_type="stock", code="OLD", name="旧季度", nav_ratio_pct=9.0,
                    disclosure_period="2024年1季度股票投资明细",
                ),
                FundHolding(
                    asset_type="stock", code="NEW", name="新季度", nav_ratio_pct=5.0,
                    disclosure_period="2024年2季度股票投资明细",
                ),
            ],
            sources=[
                SourceRecord(
                    name="测试源",
                    endpoint="test_endpoint",
                    url="https://example.com",
                    retrieved_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    as_of=date(2024, 12, 31),
                )
            ],
            warnings=["持仓为定期披露。"],
        )


class CapturingNarrator:
    def __init__(self):
        self.evidence = None

    def generate(self, evidence):
        self.evidence = evidence
        return "## 一句话结论\n仅作研究说明。"


class CapturingDebate:
    def __init__(self):
        self.evidence = None

    def run(self, evidence):
        self.evidence = evidence
        return ForecastDebate(
            status="complete",
            review=DebateReview(passed=True),
            adjudication=ForecastAdjudication(
                stance="uncertain",
                confidence="low",
                summary="样本有限，保留不确定性。",
            ),
        )


class ChinaFundAdvisorTests(unittest.TestCase):
    def test_ai_receives_compact_validated_evidence(self):
        narrator = CapturingNarrator()
        advisor = ChinaFundAdvisor(FakeDataSource(), narrator=narrator)

        analysis = advisor.analyze("000001", as_of=date(2025, 1, 1))

        self.assertIn("calculated_metrics", narrator.evidence)
        self.assertNotIn("nav_history", narrator.evidence)
        self.assertEqual(narrator.evidence["latest_top10_stock_nav_ratio_pct"], 5.0)
        self.assertEqual(
            narrator.evidence["latest_top_stock_holdings_disclosed"][0]["code"], "NEW"
        )
        self.assertEqual(analysis.ai_analysis, "## 一句话结论\n仅作研究说明。")

    def test_markdown_separates_calculation_from_ai(self):
        advisor = ChinaFundAdvisor(FakeDataSource(), narrator=CapturingNarrator())
        report = render_markdown(advisor.analyze("000001"))

        self.assertIn("代码计算的历史指标", report)
        self.assertIn("DeepSeek 证据解释", report)
        self.assertIn("不构成基金销售", report)
        self.assertIn("内容仅供研究参考，不构成投资建议。", report)
        self.assertEqual(report.count("内容仅供研究参考，不构成投资建议。"), 1)
        self.assertIn("test_endpoint", report)

    def test_prediction_builds_evidence_records_and_renders_review(self):
        debate = CapturingDebate()
        advisor = ChinaFundAdvisor(FakeDataSource(), debate_orchestrator=debate)

        analysis = advisor.predict("000001")
        report = render_markdown(analysis)

        evidence_ids = {item["id"] for item in debate.evidence}
        self.assertIn("FORECAST_H20", evidence_ids)
        self.assertIn("RISK_METRICS", evidence_ids)
        stock_evidence = next(
            item for item in debate.evidence if item["id"] == "STOCK_HOLDINGS"
        )
        self.assertEqual(stock_evidence["top10_nav_ratio_pct"], 5.0)
        self.assertIn("Python 基础概率预测", report)
        self.assertIn("多 Agent 辩论与审查", report)
        self.assertIn("概率调整：未调整", report)
        self.assertEqual(report.count("内容仅供研究参考，不构成投资建议。"), 1)

    def test_unsafe_trade_instruction_is_rejected(self):
        class UnsafeNarrator:
            def generate(self, evidence):
                return "建议立即买入。"

        advisor = ChinaFundAdvisor(FakeDataSource(), narrator=UnsafeNarrator())

        with self.assertRaisesRegex(RuntimeError, "交易指令"):
            advisor.analyze("000001")


if __name__ == "__main__":
    unittest.main()
