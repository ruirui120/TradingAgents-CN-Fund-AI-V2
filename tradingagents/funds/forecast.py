"""Transparent historical-scenario forecasts for China public funds."""

from __future__ import annotations

import heapq
import math
import statistics
from datetime import date
from typing import Iterable

from .models import (
    DirectionProbabilities,
    DrawdownProbabilities,
    ForecastBacktest,
    FundForecast,
    HorizonForecast,
    NavPoint,
)


DEFAULT_HORIZONS = (20, 60, 250)
MIN_SCENARIOS = 60
MIN_RETURNS_FOR_VOLATILITY = 30
FEATURE_LOOKBACK = 60
MAX_ANALOG_SCENARIOS = 200
MAX_BACKTEST_POINTS = 80
EQUAL_PROBABILITY_BRIER = 0.6667
MIN_MATERIAL_BRIER_IMPROVEMENT = 0.02
MIN_UNCONDITIONAL_BRIER_IMPROVEMENT = 0.01


def calculate_fund_forecast(
    nav_points: Iterable[NavPoint],
    as_of: date | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> FundForecast:
    """Estimate forward scenarios using only observations available by ``as_of``.

    This is an empirical rolling-window distribution, not a causal price model.
    It deliberately reports insufficient data instead of extrapolating.
    """

    points_by_date = {
        point.nav_date: point
        for point in nav_points
        if as_of is None or point.nav_date <= as_of
    }
    points = [points_by_date[key] for key in sorted(points_by_date)]
    forecast_date = points[-1].nav_date if points else (as_of or date.today())
    warnings = [
        "概率和区间来自历史滚动情景频率，不是对未来收益的保证。",
        "历史结构、基金经理、持仓和市场环境变化可能使历史情景失效。",
        "V2 的基础概率未经多 Agent 主观修改；审议只解释、质疑并降低结论置信度。",
    ]

    if len(points) < 2:
        return FundForecast(
            as_of=forecast_date,
            method="历史滚动情景分布",
            horizons=[_insufficient(horizon, 0) for horizon in horizons],
            warnings=warnings + ["净值记录少于 2 条，所有预测周期均不可用。"],
        )

    wealth, returns = _wealth_index(points)
    results = [
        _forecast_horizon(wealth, returns, horizon)
        for horizon in horizons
    ]
    return FundForecast(
        as_of=forecast_date,
        method=(
            "历史相似情景分布（匹配 20/60 日趋势和 20 日波动，"
            "严格按截止日截断，非因果预测模型）"
        ),
        horizons=results,
        warnings=warnings,
    )


def _forecast_horizon(
    wealth: list[float], returns: list[float], horizon: int
) -> HorizonForecast:
    current_origin = len(wealth) - 1
    analog_returns, analog_drawdowns, candidate_count, all_returns, all_drawdowns = _historical_analogs(
        wealth, returns, horizon, current_origin
    )
    analog_count = len(analog_returns)
    if analog_count < MIN_SCENARIOS or len(returns) < MIN_RETURNS_FOR_VOLATILITY:
        result = _insufficient(horizon, candidate_count, analog_count)
        result.notes.append(
            f"至少需要 {MIN_SCENARIOS} 个完整历史窗口和 "
            f"{MIN_RETURNS_FOR_VOLATILITY} 条日收益，本周期不强行外推。"
        )
        return result

    backtest = _walk_forward_backtest(wealth, returns, horizon)
    use_analogs = bool(
        candidate_count >= 250
        and backtest.sample_count >= 60
        and backtest.materially_beats_equal_probability_baseline
        and backtest.materially_beats_unconditional_baseline
    )
    selected_model = "similar_scenarios" if use_analogs else "unconditional_history"
    forward_returns = analog_returns if use_analogs else all_returns
    forward_drawdowns = analog_drawdowns if use_analogs else all_drawdowns
    scenario_count = len(forward_returns)
    neutral_band = _neutral_band(returns, horizon)
    probabilities = _direction_probabilities(forward_returns, neutral_band)
    model_note = (
        "相似情景模型通过样本外增量门槛，本周期使用当前状态相似情景。"
        if use_analogs
        else "相似情景模型未通过样本外增量门槛，本周期回退到全历史无条件情景。"
    )
    return HorizonForecast(
        horizon_days=horizon,
        available=True,
        selected_model=selected_model,
        candidate_count=candidate_count,
        analog_scenario_count=analog_count,
        scenario_count=scenario_count,
        reliability="medium" if use_analogs else "low",
        neutral_band_pct=_pct(neutral_band),
        probabilities=probabilities,
        return_p10_pct=_pct(_percentile(forward_returns, 0.10)),
        return_p50_pct=_pct(_percentile(forward_returns, 0.50)),
        return_p90_pct=_pct(_percentile(forward_returns, 0.90)),
        loss_probability_pct=_frequency(item < 0 for item in forward_returns),
        drawdown_probabilities=DrawdownProbabilities(
            over_10_pct=_frequency(item <= -0.10 for item in forward_drawdowns),
            over_15_pct=_frequency(item <= -0.15 for item in forward_drawdowns),
            over_20_pct=_frequency(item <= -0.20 for item in forward_drawdowns),
        ),
        backtest=backtest,
        notes=[
            model_note,
            "震荡区间随历史波动和预测周期调整，避免把微小变化误报为明确涨跌。",
            "相似情景按当前 20/60 日趋势与 20 日波动的标准化距离选择。",
            "历史窗口彼此重叠，情景数量不等于独立样本数量。",
            "样本外检验在全历史均匀抽取最多 80 个时点，并只使用当时已经完成的历史情景。",
        ],
    )


def _insufficient(
    horizon: int, candidate_count: int, scenario_count: int = 0
) -> HorizonForecast:
    return HorizonForecast(
        horizon_days=horizon,
        available=False,
        selected_model="unavailable",
        candidate_count=candidate_count,
        analog_scenario_count=scenario_count,
        scenario_count=scenario_count,
        reliability="insufficient",
        notes=["历史样本不足，未生成方向概率、收益区间或回撤概率。"],
    )


def _wealth_index(points: list[NavPoint]) -> tuple[list[float], list[float]]:
    published = [point.daily_return_pct for point in points[1:]]
    if all(value is not None and value > -100 for value in published):
        returns = [float(value) / 100 for value in published]
        wealth = [1.0]
        for daily_return in returns:
            wealth.append(wealth[-1] * (1 + daily_return))
        return wealth, returns

    use_accumulated = all(point.accumulated_nav is not None for point in points)
    wealth = [
        float(point.accumulated_nav if use_accumulated else point.unit_nav)
        for point in points
    ]
    returns = [current / previous - 1 for previous, current in zip(wealth, wealth[1:])]
    return wealth, returns


def _neutral_band(returns: list[float], horizon: int) -> float:
    daily_volatility = statistics.stdev(returns)
    scaled_band = daily_volatility * math.sqrt(horizon) * 0.25
    return min(0.05, max(0.01, scaled_band))


def _historical_analogs(
    wealth: list[float],
    returns: list[float],
    horizon: int,
    current_origin: int,
) -> tuple[list[float], list[float], int, list[float], list[float]]:
    latest_candidate_origin = current_origin - horizon
    if latest_candidate_origin < FEATURE_LOOKBACK:
        return [], [], 0, [], []

    current_features = _feature_vector(wealth, returns, current_origin)
    candidates = []
    for origin in range(FEATURE_LOOKBACK, latest_candidate_origin + 1):
        features = _feature_vector(wealth, returns, origin)
        forward_return = wealth[origin + horizon] / wealth[origin] - 1
        drawdown = _path_max_drawdown(wealth[origin : origin + horizon + 1])
        candidates.append((features, forward_return, drawdown))

    scales = []
    for feature_index in range(3):
        values = [item[0][feature_index] for item in candidates]
        scale = statistics.pstdev(values)
        scales.append(scale if scale > 0 else 1.0)

    selected = heapq.nsmallest(
        MAX_ANALOG_SCENARIOS,
        candidates,
        key=lambda item: sum(
            ((value - target) / scale) ** 2
            for value, target, scale in zip(item[0], current_features, scales)
        ),
    )
    return (
        [item[1] for item in selected],
        [item[2] for item in selected],
        len(candidates),
        [item[1] for item in candidates],
        [item[2] for item in candidates],
    )


def _feature_vector(
    wealth: list[float], returns: list[float], origin: int
) -> tuple[float, float, float]:
    momentum_20 = wealth[origin] / wealth[origin - 20] - 1
    momentum_60 = wealth[origin] / wealth[origin - 60] - 1
    volatility_20 = statistics.stdev(returns[origin - 20 : origin])
    return momentum_20, momentum_60, volatility_20


def _direction_probabilities(
    forward_returns: list[float], neutral_band: float
) -> DirectionProbabilities:
    upward = _frequency(item > neutral_band for item in forward_returns)
    downward = _frequency(item < -neutral_band for item in forward_returns)
    sideways = round(100 - upward - downward, 4)
    return DirectionProbabilities(
        upward_pct=upward,
        sideways_pct=sideways,
        downward_pct=downward,
    )


def _walk_forward_backtest(
    wealth: list[float], returns: list[float], horizon: int
) -> ForecastBacktest:
    scores: list[float] = []
    unconditional_scores: list[float] = []
    correct = 0
    first_origin = horizon + FEATURE_LOOKBACK + MIN_SCENARIOS - 1
    available_origins = max(0, len(wealth) - horizon - first_origin)
    step = max(1, math.ceil(available_origins / MAX_BACKTEST_POINTS))
    for origin in range(first_origin, len(wealth) - horizon, step):
        historical, _, _, unconditional, _ = _historical_analogs(
            wealth, returns, horizon, origin
        )
        if len(historical) < MIN_SCENARIOS:
            continue
        band = _neutral_band(returns[:origin], horizon)
        probabilities = _direction_probabilities(historical, band)
        predicted = [
            probabilities.upward_pct / 100,
            probabilities.sideways_pct / 100,
            probabilities.downward_pct / 100,
        ]
        realized_return = wealth[origin + horizon] / wealth[origin] - 1
        actual_index = _direction_index(realized_return, band)
        actual = [1.0 if index == actual_index else 0.0 for index in range(3)]
        scores.append(
            sum(
                (estimate - outcome) ** 2
                for estimate, outcome in zip(predicted, actual)
            )
        )
        unconditional_probabilities = _direction_probabilities(unconditional, band)
        unconditional_estimate = [
            unconditional_probabilities.upward_pct / 100,
            unconditional_probabilities.sideways_pct / 100,
            unconditional_probabilities.downward_pct / 100,
        ]
        unconditional_scores.append(
            sum(
                (estimate - outcome) ** 2
                for estimate, outcome in zip(unconditional_estimate, actual)
            )
        )
        if predicted.index(max(predicted)) == actual_index:
            correct += 1

    if not scores:
        return ForecastBacktest(sample_count=0)
    brier_score = round(statistics.fmean(scores), 4)
    brier_improvement = round(EQUAL_PROBABILITY_BRIER - brier_score, 4)
    unconditional_brier = round(statistics.fmean(unconditional_scores), 4)
    unconditional_improvement = round(unconditional_brier - brier_score, 4)
    return ForecastBacktest(
        sample_count=len(scores),
        brier_score=brier_score,
        equal_probability_brier_score=EQUAL_PROBABILITY_BRIER,
        brier_improvement=brier_improvement,
        materially_beats_equal_probability_baseline=(
            brier_improvement >= MIN_MATERIAL_BRIER_IMPROVEMENT
        ),
        unconditional_brier_score=unconditional_brier,
        brier_improvement_vs_unconditional=unconditional_improvement,
        materially_beats_unconditional_baseline=(
            unconditional_improvement >= MIN_UNCONDITIONAL_BRIER_IMPROVEMENT
        ),
        most_likely_direction_accuracy_pct=round(correct / len(scores) * 100, 2),
    )


def _direction_index(value: float, neutral_band: float) -> int:
    if value > neutral_band:
        return 0
    if value < -neutral_band:
        return 2
    return 1


def _path_max_drawdown(values: list[float]) -> float:
    peak = values[0]
    worst = 0.0
    for value in values[1:]:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1)
    return worst


def _frequency(flags) -> float:
    values = list(flags)
    return round(sum(values) / len(values) * 100, 4) if values else 0.0


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pct(value: float) -> float:
    return round(value * 100, 4)
