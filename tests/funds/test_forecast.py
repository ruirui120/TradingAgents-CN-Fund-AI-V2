from datetime import date, timedelta
import unittest

from tradingagents.funds.forecast import calculate_fund_forecast
from tradingagents.funds.models import NavPoint


def synthetic_points(count: int) -> list[NavPoint]:
    points = []
    nav = 1.0
    pattern = [0.35, -0.20, 0.15, 0.05, -0.10, 0.25, -0.05]
    for index in range(count):
        daily_return_pct = 0.0 if index == 0 else pattern[index % len(pattern)]
        nav *= 1 + daily_return_pct / 100
        points.append(
            NavPoint(
                nav_date=date(2020, 1, 1) + timedelta(days=index),
                unit_nav=nav,
                daily_return_pct=daily_return_pct,
            )
        )
    return points


def regime_points(final_daily_return_pct: float) -> list[NavPoint]:
    returns = []
    for _ in range(5):
        returns.extend([0.25] * 60 + [0.45] * 20)
        returns.extend([-0.25] * 60 + [-0.45] * 20)
    returns.extend([final_daily_return_pct] * 60)
    nav = 1.0
    points = []
    for index, daily_return_pct in enumerate([0.0] + returns):
        nav *= 1 + daily_return_pct / 100
        points.append(
            NavPoint(
                nav_date=date(2020, 1, 1) + timedelta(days=index),
                unit_nav=nav,
                daily_return_pct=daily_return_pct,
            )
        )
    return points


class FundForecastTests(unittest.TestCase):
    def test_forecast_probabilities_and_intervals_are_internally_consistent(self):
        forecast = calculate_fund_forecast(synthetic_points(900))

        for result in forecast.horizons:
            self.assertTrue(result.available)
            probabilities = result.probabilities
            self.assertAlmostEqual(
                probabilities.upward_pct
                + probabilities.sideways_pct
                + probabilities.downward_pct,
                100,
            )
            self.assertLessEqual(result.return_p10_pct, result.return_p50_pct)
            self.assertLessEqual(result.return_p50_pct, result.return_p90_pct)
            self.assertGreaterEqual(
                result.drawdown_probabilities.over_10_pct,
                result.drawdown_probabilities.over_15_pct,
            )
            self.assertGreaterEqual(
                result.drawdown_probabilities.over_15_pct,
                result.drawdown_probabilities.over_20_pct,
            )
            self.assertGreater(result.backtest.sample_count, 0)
            self.assertIsNotNone(
                result.backtest.materially_beats_equal_probability_baseline
            )
            self.assertIsNotNone(
                result.backtest.materially_beats_unconditional_baseline
            )
            if result.reliability == "medium":
                self.assertEqual(result.selected_model, "similar_scenarios")
                self.assertTrue(
                    result.backtest.materially_beats_equal_probability_baseline
                )
                self.assertTrue(
                    result.backtest.materially_beats_unconditional_baseline
                )
            else:
                self.assertEqual(result.selected_model, "unconditional_history")

    def test_as_of_cutoff_prevents_future_data_leakage(self):
        points = synthetic_points(700)
        cutoff = points[599].nav_date
        original = calculate_fund_forecast(points[:600], as_of=cutoff)

        future = points[600:]
        for point in future:
            point.daily_return_pct = 25.0
        with_future_attached = calculate_fund_forecast(points, as_of=cutoff)

        self.assertEqual(original, with_future_attached)

    def test_unavailable_horizon_does_not_invent_numbers(self):
        forecast = calculate_fund_forecast(synthetic_points(200))
        by_horizon = {item.horizon_days: item for item in forecast.horizons}

        self.assertTrue(by_horizon[20].available)
        self.assertTrue(by_horizon[60].available)
        self.assertFalse(by_horizon[250].available)
        self.assertIsNone(by_horizon[250].probabilities)
        self.assertIsNone(by_horizon[250].return_p50_pct)
        self.assertEqual(by_horizon[250].reliability, "insufficient")
        self.assertEqual(by_horizon[250].selected_model, "unavailable")

    def test_forecast_changes_when_current_trend_regime_changes(self):
        upward_regime = calculate_fund_forecast(
            regime_points(0.25), horizons=(20,)
        ).horizons[0]
        downward_regime = calculate_fund_forecast(
            regime_points(-0.25), horizons=(20,)
        ).horizons[0]

        self.assertNotEqual(upward_regime.probabilities, downward_regime.probabilities)


if __name__ == "__main__":
    unittest.main()
