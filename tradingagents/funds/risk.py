"""Deterministic risk calculations for daily China fund NAV data."""

from __future__ import annotations

import math
import statistics
from typing import Iterable, Optional

from .models import FundRiskMetrics, NavPoint


TRADING_DAYS_PER_YEAR = 250
MIN_RETURN_OBSERVATIONS = 30
MIN_CALENDAR_DAYS_FOR_ANNUALIZATION = 180


def calculate_risk_metrics(
    nav_points: Iterable[NavPoint],
    annual_risk_free_rate: Optional[float] = None,
) -> FundRiskMetrics:
    """Calculate reproducible metrics without asking an LLM to do arithmetic.

    ``annual_risk_free_rate`` is a decimal (for example, 0.015 for 1.5%).
    Sharpe is omitted when the caller does not provide this assumption.
    """

    points_by_date = {point.nav_date: point for point in nav_points}
    points = [points_by_date[key] for key in sorted(points_by_date)]
    notes: list[str] = [
        f"波动率按每年 {TRADING_DAYS_PER_YEAR} 个交易日年化。",
        "最大回撤基于同一收益口径重建的历史财富指数计算。",
        "历史 VaR(95%) 为历史模拟法的一日损失分位数，不代表最大可能损失。",
    ]

    if len(points) < 2:
        return FundRiskMetrics(
            observation_count=len(points),
            start_date=points[0].nav_date if points else None,
            end_date=points[-1].nav_date if points else None,
            notes=notes + ["净值记录少于 2 条，无法计算收益和风险指标。"],
        )

    published_returns = [point.daily_return_pct for point in points[1:]]
    use_published_returns = all(
        value is not None and value > -100 for value in published_returns
    )
    if use_published_returns:
        returns = [value / 100 for value in published_returns]
        nav_values = [1.0]
        for daily_return in returns:
            nav_values.append(nav_values[-1] * (1 + daily_return))
        notes.append(
            "收益与回撤使用基金公布的每日增长率复利重建，降低分红和份额折算造成的口径失真。"
        )
    else:
        use_accumulated_nav = all(point.accumulated_nav is not None for point in points)
        nav_values = [
            point.accumulated_nav if use_accumulated_nav else point.unit_nav
            for point in points
        ]
        returns = [
            current / previous - 1
            for previous, current in zip(nav_values, nav_values[1:])
            if previous > 0
        ]
        notes.append(
            "每日增长率不完整，改用累计净值；累计净值也不完整时退回单位净值，"
            "分红或份额折算可能使收益与回撤失真。"
        )
    start_date = points[0].nav_date
    end_date = points[-1].nav_date
    total_return = nav_values[-1] / nav_values[0] - 1

    annualized_return: Optional[float] = None
    calendar_days = (end_date - start_date).days
    if calendar_days >= MIN_CALENDAR_DAYS_FOR_ANNUALIZATION and 1 + total_return > 0:
        annualized_return = (1 + total_return) ** (365.25 / calendar_days) - 1
    else:
        notes.append(
            f"历史跨度不足 {MIN_CALENDAR_DAYS_FOR_ANNUALIZATION} 天，未展示容易误导的年化收益。"
        )

    volatility: Optional[float] = None
    historical_var: Optional[float] = None
    positive_ratio: Optional[float] = None
    sharpe: Optional[float] = None
    if len(returns) >= MIN_RETURN_OBSERVATIONS:
        daily_std = statistics.stdev(returns)
        volatility = daily_std * math.sqrt(TRADING_DAYS_PER_YEAR)
        historical_var = max(0.0, -_percentile(returns, 0.05))
        positive_ratio = sum(item > 0 for item in returns) / len(returns)
        if annual_risk_free_rate is not None and volatility > 0:
            annualized_mean_return = statistics.fmean(returns) * TRADING_DAYS_PER_YEAR
            sharpe = (annualized_mean_return - annual_risk_free_rate) / volatility
    else:
        notes.append(
            f"有效日收益少于 {MIN_RETURN_OBSERVATIONS} 条，未展示波动率、VaR 和上涨日比例。"
        )

    if annual_risk_free_rate is None:
        notes.append("未提供无风险利率假设，因此不展示夏普比率。")

    max_drawdown, peak_date, trough_date = _max_drawdown(points, nav_values)
    return FundRiskMetrics(
        observation_count=len(points),
        start_date=start_date,
        end_date=end_date,
        total_return_pct=_pct(total_return),
        annualized_return_pct=_pct(annualized_return),
        annualized_volatility_pct=_pct(volatility),
        max_drawdown_pct=_pct(max_drawdown),
        max_drawdown_peak_date=peak_date,
        max_drawdown_trough_date=trough_date,
        historical_var_95_pct=_pct(historical_var),
        positive_day_ratio_pct=_pct(positive_ratio),
        sharpe_ratio=_round(sharpe),
        annual_risk_free_rate_pct=_pct(annual_risk_free_rate),
        notes=notes,
    )


def _max_drawdown(points: list[NavPoint], nav_values: list[float]):
    peak_value = nav_values[0]
    peak_date = points[0].nav_date
    worst_drawdown = 0.0
    worst_peak = peak_date
    worst_trough = peak_date

    for point, nav_value in zip(points[1:], nav_values[1:]):
        if nav_value > peak_value:
            peak_value = nav_value
            peak_date = point.nav_date
        drawdown = nav_value / peak_value - 1
        if drawdown < worst_drawdown:
            worst_drawdown = drawdown
            worst_peak = peak_date
            worst_trough = point.nav_date
    return worst_drawdown, worst_peak, worst_trough


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _pct(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value * 100, 4)


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)
