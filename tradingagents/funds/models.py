"""Typed data contracts for China public mutual fund research."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class SourceRecord(BaseModel):
    """Provenance for one externally retrieved dataset."""

    name: str
    endpoint: str
    url: str
    retrieved_at: datetime
    as_of: Optional[date] = None
    note: Optional[str] = None


class FundProfile(BaseModel):
    code: str
    name: str
    full_name: Optional[str] = None
    category: Optional[str] = None
    investment_type: Optional[str] = None
    manager_names: list[str] = Field(default_factory=list)
    company: Optional[str] = None
    custodian: Optional[str] = None
    inception_date: Optional[date] = None
    scale_text: Optional[str] = None
    benchmark: Optional[str] = None
    strategy: Optional[str] = None
    objective: Optional[str] = None
    management_fee: Optional[str] = None
    custody_fee: Optional[str] = None
    max_purchase_fee: Optional[str] = None
    max_redemption_fee: Optional[str] = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        normalized = str(value).strip()
        if len(normalized) != 6 or not normalized.isdigit():
            raise ValueError("中国公募基金代码必须是 6 位数字")
        return normalized


class NavPoint(BaseModel):
    nav_date: date
    unit_nav: float = Field(gt=0)
    accumulated_nav: Optional[float] = Field(default=None, gt=0)
    daily_return_pct: Optional[float] = None


class FundHolding(BaseModel):
    asset_type: Literal["stock", "bond"]
    code: str
    name: str
    nav_ratio_pct: Optional[float] = Field(default=None, ge=0)
    market_value_ten_thousand_cny: Optional[float] = Field(default=None, ge=0)
    disclosure_period: Optional[str] = None


class IndustryAllocation(BaseModel):
    industry: str
    nav_ratio_pct: Optional[float] = Field(default=None, ge=0)
    market_value_ten_thousand_cny: Optional[float] = Field(default=None, ge=0)
    disclosure_date: Optional[date] = None


class FundDataset(BaseModel):
    profile: FundProfile
    nav_history: list[NavPoint]
    holdings: list[FundHolding] = Field(default_factory=list)
    industry_allocations: list[IndustryAllocation] = Field(default_factory=list)
    sources: list[SourceRecord]
    warnings: list[str] = Field(default_factory=list)


class FundRiskMetrics(BaseModel):
    observation_count: int
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    total_return_pct: Optional[float] = None
    annualized_return_pct: Optional[float] = None
    annualized_volatility_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    max_drawdown_peak_date: Optional[date] = None
    max_drawdown_trough_date: Optional[date] = None
    historical_var_95_pct: Optional[float] = None
    positive_day_ratio_pct: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    annual_risk_free_rate_pct: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


class DirectionProbabilities(BaseModel):
    upward_pct: float = Field(ge=0, le=100)
    sideways_pct: float = Field(ge=0, le=100)
    downward_pct: float = Field(ge=0, le=100)

    @field_validator("downward_pct")
    @classmethod
    def validate_total(cls, value: float, info) -> float:
        upward = info.data.get("upward_pct", 0)
        sideways = info.data.get("sideways_pct", 0)
        if abs(upward + sideways + value - 100) > 0.05:
            raise ValueError("方向概率之和必须为 100%")
        return value


class DrawdownProbabilities(BaseModel):
    over_10_pct: float = Field(ge=0, le=100)
    over_15_pct: float = Field(ge=0, le=100)
    over_20_pct: float = Field(ge=0, le=100)


class ForecastBacktest(BaseModel):
    sample_count: int = Field(ge=0)
    brier_score: Optional[float] = Field(default=None, ge=0, le=2)
    equal_probability_brier_score: float = 0.6667
    brier_improvement: Optional[float] = None
    materially_beats_equal_probability_baseline: Optional[bool] = None
    unconditional_brier_score: Optional[float] = Field(default=None, ge=0, le=2)
    brier_improvement_vs_unconditional: Optional[float] = None
    materially_beats_unconditional_baseline: Optional[bool] = None
    most_likely_direction_accuracy_pct: Optional[float] = Field(
        default=None, ge=0, le=100
    )


class HorizonForecast(BaseModel):
    horizon_days: int = Field(gt=0)
    available: bool
    selected_model: Literal["unavailable", "similar_scenarios", "unconditional_history"]
    candidate_count: int = Field(ge=0)
    analog_scenario_count: int = Field(ge=0)
    scenario_count: int = Field(ge=0)
    reliability: Literal["insufficient", "low", "medium"]
    neutral_band_pct: Optional[float] = Field(default=None, ge=0)
    probabilities: Optional[DirectionProbabilities] = None
    return_p10_pct: Optional[float] = None
    return_p50_pct: Optional[float] = None
    return_p90_pct: Optional[float] = None
    loss_probability_pct: Optional[float] = Field(default=None, ge=0, le=100)
    drawdown_probabilities: Optional[DrawdownProbabilities] = None
    backtest: Optional[ForecastBacktest] = None
    notes: list[str] = Field(default_factory=list)


class FundForecast(BaseModel):
    as_of: date
    method: str
    horizons: list[HorizonForecast]
    warnings: list[str] = Field(default_factory=list)


class EvidenceClaim(BaseModel):
    statement: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1)


class AgentOpinion(BaseModel):
    role: str
    stance: Literal["bullish", "neutral", "bearish", "uncertain"]
    confidence: float = Field(ge=0, le=1)
    claims: list[EvidenceClaim] = Field(default_factory=list)
    risks: list[EvidenceClaim] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class AgentRebuttal(BaseModel):
    role: Literal["bull", "bear"]
    challenges: list[EvidenceClaim] = Field(default_factory=list)
    concessions: list[str] = Field(default_factory=list)


class DebateReview(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    accepted_evidence_ids: list[str] = Field(default_factory=list)
    rejected_items: list[str] = Field(default_factory=list)
    duplicate_evidence_groups: list[list[str]] = Field(default_factory=list)


class ForecastAdjudication(BaseModel):
    stance: Literal["bullish", "neutral", "bearish", "uncertain"]
    confidence: Literal["low", "medium"]
    summary: str
    consensus: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    probability_adjustment_applied: bool = False
    adjustment_reason: str = (
        "V2 尚未完成样本外校准，Agent 审议不得修改 Python 生成的基础概率。"
    )


class ForecastDebate(BaseModel):
    status: Literal["complete", "degraded", "unavailable"]
    opinions: list[AgentOpinion] = Field(default_factory=list)
    rebuttals: list[AgentRebuttal] = Field(default_factory=list)
    review: Optional[DebateReview] = None
    adjudication: Optional[ForecastAdjudication] = None
    failures: list[str] = Field(default_factory=list)


class FundAnalysis(BaseModel):
    dataset: FundDataset
    metrics: FundRiskMetrics
    ai_analysis: Optional[str] = None
    forecast: Optional[FundForecast] = None
    debate: Optional[ForecastDebate] = None
    generated_at: datetime
    purpose: str = "研究与投资者教育"
    disclaimer: str = (
        "内容仅供研究参考，不构成投资建议。"
        "本报告基于公开历史数据生成，不构成基金销售、收益预测或个性化投资建议；"
        "基金过往业绩不预示未来表现，投资决策与损失由投资者自行承担。"
    )
