from datetime import date, timedelta
import unittest

from tradingagents.funds.models import NavPoint
from tradingagents.funds.risk import calculate_risk_metrics


class FundRiskMetricsTests(unittest.TestCase):
    def test_max_drawdown_uses_prior_peak(self):
        values = [1.0, 1.2, 0.9, 1.0]
        points = [
            NavPoint(nav_date=date(2025, 1, 1) + timedelta(days=index), unit_nav=value)
            for index, value in enumerate(values)
        ]

        metrics = calculate_risk_metrics(points)

        self.assertEqual(metrics.max_drawdown_pct, -25.0)
        self.assertEqual(metrics.max_drawdown_peak_date, date(2025, 1, 2))
        self.assertEqual(metrics.max_drawdown_trough_date, date(2025, 1, 3))
        self.assertIsNone(metrics.annualized_volatility_pct)

    def test_annualized_metrics_require_enough_history(self):
        points = [
            NavPoint(
                nav_date=date(2024, 1, 1) + timedelta(days=index * 7),
                unit_nav=1 + index * 0.002,
            )
            for index in range(60)
        ]

        metrics = calculate_risk_metrics(points, annual_risk_free_rate=0.015)

        self.assertEqual(metrics.observation_count, 60)
        self.assertIsNotNone(metrics.annualized_return_pct)
        self.assertIsNotNone(metrics.annualized_volatility_pct)
        self.assertIsNotNone(metrics.historical_var_95_pct)
        self.assertIsNotNone(metrics.sharpe_ratio)
        self.assertEqual(metrics.annual_risk_free_rate_pct, 1.5)

    def test_sharpe_is_omitted_without_explicit_risk_free_rate(self):
        points = [
            NavPoint(
                nav_date=date(2025, 1, 1) + timedelta(days=index),
                unit_nav=1 + index * 0.001,
            )
            for index in range(40)
        ]

        metrics = calculate_risk_metrics(points)

        self.assertIsNone(metrics.sharpe_ratio)
        self.assertTrue(any("无风险利率" in note for note in metrics.notes))

    def test_accumulated_nav_is_used_when_available(self):
        points = [
            NavPoint(nav_date=date(2025, 1, 1), unit_nav=1.0, accumulated_nav=1.0),
            NavPoint(nav_date=date(2025, 2, 1), unit_nav=0.9, accumulated_nav=1.1),
        ]

        metrics = calculate_risk_metrics(points)

        self.assertEqual(metrics.total_return_pct, 10.0)
        self.assertEqual(metrics.max_drawdown_pct, 0.0)
        self.assertTrue(any("每日增长率不完整" in note for note in metrics.notes))

    def test_published_daily_return_rebuilds_total_return_and_drawdown(self):
        points = [
            NavPoint(nav_date=date(2025, 1, 1), unit_nav=1.0, daily_return_pct=0.0),
            NavPoint(nav_date=date(2025, 1, 2), unit_nav=0.9, daily_return_pct=10.0),
            NavPoint(nav_date=date(2025, 1, 3), unit_nav=1.1, daily_return_pct=-20.0),
        ]

        metrics = calculate_risk_metrics(points)

        self.assertEqual(metrics.total_return_pct, -12.0)
        self.assertEqual(metrics.max_drawdown_pct, -20.0)
        self.assertTrue(any("每日增长率复利重建" in note for note in metrics.notes))


if __name__ == "__main__":
    unittest.main()
