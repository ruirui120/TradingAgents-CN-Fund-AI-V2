"""AKShare-backed public mutual fund data source."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from ..models import (
    FundDataset,
    FundHolding,
    FundProfile,
    IndustryAllocation,
    NavPoint,
    SourceRecord,
)


AKSHARE_FUND_DOCS = "https://akshare.akfamily.xyz/data/fund/fund_public.html"


class FundDataError(RuntimeError):
    """Raised when public fund data cannot be validated."""


class AkshareFundDataSource:
    """Retrieve delayed public data through documented AKShare functions."""

    def __init__(self, akshare_module: Any = None):
        if akshare_module is None:
            try:
                import akshare as akshare_module
            except ImportError as exc:
                raise FundDataError("未安装 akshare；请先安装项目依赖。") from exc
        self.ak = akshare_module

    def fetch(self, fund_code: str, as_of: date | None = None) -> FundDataset:
        code = _normalize_code(fund_code)
        cutoff = as_of or date.today()
        retrieved_at = datetime.now(timezone.utc)
        warnings: list[str] = [
            "基金净值通常为交易日收盘后的 T+0/T+1 数据，不是盘中实时成交价。",
            "基金持仓来自定期报告，可能已滞后，不能视为当前实时持仓。",
        ]

        basic_df = self.ak.fund_individual_basic_info_xq(symbol=code)
        basic = _key_value_frame(basic_df)
        if not basic:
            basic = self._fallback_basic(code)

        profile = _build_profile(code, basic)
        if profile.category and "货币" in profile.category:
            raise FundDataError(
                "V1 暂不支持货币基金；货币基金应使用万份收益和七日年化收益率，"
                "不能套用普通基金单位净值算法。"
            )
        nav_history = self._fetch_nav(code, cutoff)
        if not nav_history:
            raise FundDataError(f"基金 {code} 没有可用的单位净值历史数据。")

        latest_nav_date = nav_history[-1].nav_date
        age_days = (cutoff - latest_nav_date).days
        if age_days > 10:
            warnings.append(f"最新净值日期为 {latest_nav_date}，距分析日 {age_days} 天，数据可能过期。")

        stock_holdings = self._fetch_holdings(
            self.ak.fund_portfolio_hold_em,
            code,
            cutoff,
            asset_type="stock",
        )
        bond_holdings = self._fetch_holdings(
            self.ak.fund_portfolio_bond_hold_em,
            code,
            cutoff,
            asset_type="bond",
        )
        industries = self._fetch_industries(code, cutoff)

        sources = [
            SourceRecord(
                name="雪球基金（经 AKShare）",
                endpoint="fund_individual_basic_info_xq",
                url=AKSHARE_FUND_DOCS,
                retrieved_at=retrieved_at,
                note="基金基本信息；第三方公开页面，字段可能调整。",
            ),
            SourceRecord(
                name="天天基金/东方财富（经 AKShare）",
                endpoint="fund_open_fund_info_em",
                url=AKSHARE_FUND_DOCS,
                retrieved_at=retrieved_at,
                as_of=latest_nav_date,
                note="单位净值历史。",
            ),
        ]
        if stock_holdings:
            sources.append(_holding_source("fund_portfolio_hold_em", retrieved_at, stock_holdings))
        if bond_holdings:
            sources.append(_holding_source("fund_portfolio_bond_hold_em", retrieved_at, bond_holdings))
        if industries:
            sources.append(
                SourceRecord(
                    name="天天基金/东方财富（经 AKShare）",
                    endpoint="fund_portfolio_industry_allocation_em",
                    url=AKSHARE_FUND_DOCS,
                    retrieved_at=retrieved_at,
                    as_of=max(item.disclosure_date for item in industries if item.disclosure_date),
                    note="定期报告行业配置，不代表当前实时配置。",
                )
            )

        if not stock_holdings and not bond_holdings:
            warnings.append("未取得股票或债券持仓；本次报告不能分析底层资产集中度。")

        return FundDataset(
            profile=profile,
            nav_history=nav_history,
            holdings=stock_holdings + bond_holdings,
            industry_allocations=industries,
            sources=sources,
            warnings=warnings,
        )

    def _fallback_basic(self, code: str) -> dict[str, Any]:
        frame = self.ak.fund_name_em()
        if frame is None or frame.empty or "基金代码" not in frame.columns:
            raise FundDataError(f"无法确认基金代码 {code}。")
        matches = frame[frame["基金代码"].astype(str).str.zfill(6) == code]
        if matches.empty:
            raise FundDataError(f"未找到基金代码 {code}。")
        row = matches.iloc[0]
        return {
            "基金代码": code,
            "基金名称": row.get("基金简称") or row.get("基金名称"),
            "基金类型": row.get("基金类型"),
        }

    def _fetch_nav(self, code: str, cutoff: date) -> list[NavPoint]:
        frame = self.ak.fund_open_fund_info_em(symbol=code, indicator="单位净值走势")
        required = {"净值日期", "单位净值"}
        if frame is None or frame.empty or not required.issubset(frame.columns):
            return []
        normalized = frame.copy()
        normalized["净值日期"] = pd.to_datetime(normalized["净值日期"], errors="coerce").dt.date
        normalized["单位净值"] = pd.to_numeric(normalized["单位净值"], errors="coerce")
        normalized = normalized.dropna(subset=["净值日期", "单位净值"])
        normalized = normalized[(normalized["净值日期"] <= cutoff) & (normalized["单位净值"] > 0)]
        normalized = normalized.sort_values("净值日期").drop_duplicates("净值日期", keep="last")

        accumulated = self.ak.fund_open_fund_info_em(symbol=code, indicator="累计净值走势")
        accumulated_by_date: dict[date, float] = {}
        if accumulated is not None and not accumulated.empty and {"净值日期", "累计净值"}.issubset(accumulated.columns):
            accumulated = accumulated.copy()
            accumulated["净值日期"] = pd.to_datetime(accumulated["净值日期"], errors="coerce").dt.date
            accumulated["累计净值"] = pd.to_numeric(accumulated["累计净值"], errors="coerce")
            accumulated = accumulated.dropna(subset=["净值日期", "累计净值"])
            accumulated = accumulated[(accumulated["净值日期"] <= cutoff) & (accumulated["累计净值"] > 0)]
            accumulated_by_date = {
                row["净值日期"]: float(row["累计净值"])
                for _, row in accumulated.iterrows()
            }
        return [
            NavPoint(
                nav_date=row["净值日期"],
                unit_nav=float(row["单位净值"]),
                accumulated_nav=accumulated_by_date.get(row["净值日期"]),
                daily_return_pct=_optional_float(row.get("日增长率")),
            )
            for _, row in normalized.iterrows()
        ]

    def _fetch_holdings(
        self,
        function: Callable[..., pd.DataFrame],
        code: str,
        cutoff: date,
        asset_type: str,
    ) -> list[FundHolding]:
        frame = _latest_available_year(function, code, cutoff)
        if frame is None or frame.empty:
            return []
        code_column = "股票代码" if asset_type == "stock" else "债券代码"
        name_column = "股票名称" if asset_type == "stock" else "债券名称"
        if code_column not in frame.columns or name_column not in frame.columns:
            return []
        return [
            FundHolding(
                asset_type=asset_type,
                code=str(row.get(code_column, "")).strip(),
                name=str(row.get(name_column, "")).strip(),
                nav_ratio_pct=_optional_float(row.get("占净值比例")),
                market_value_ten_thousand_cny=_non_negative_float(row.get("持仓市值")),
                disclosure_period=_optional_text(row.get("季度")),
            )
            for _, row in frame.iterrows()
            if _optional_text(row.get(name_column))
            and _period_end_date(row.get("季度"), cutoff.year) <= cutoff
        ]

    def _fetch_industries(self, code: str, cutoff: date) -> list[IndustryAllocation]:
        frame = _latest_available_year(
            self.ak.fund_portfolio_industry_allocation_em,
            code,
            cutoff,
        )
        if frame is None or frame.empty or "行业类别" not in frame.columns:
            return []
        allocations: list[IndustryAllocation] = []
        for _, row in frame.iterrows():
            industry = _optional_text(row.get("行业类别"))
            disclosure_date = _optional_date(row.get("截止时间"))
            if not industry or not disclosure_date or disclosure_date > cutoff:
                continue
            allocations.append(
                IndustryAllocation(
                    industry=industry,
                    nav_ratio_pct=_optional_float(row.get("占净值比例")),
                    market_value_ten_thousand_cny=_non_negative_float(row.get("市值")),
                    disclosure_date=disclosure_date,
                )
            )
        return allocations


def _normalize_code(value: str) -> str:
    code = str(value).strip()
    if len(code) != 6 or not code.isdigit():
        raise FundDataError("中国公募基金代码必须是 6 位数字，例如 000001。")
    return code


def _key_value_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame is None or frame.empty or len(frame.columns) < 2:
        return {}
    first, second = frame.columns[:2]
    return {
        str(row[first]).strip(): row[second]
        for _, row in frame.iterrows()
        if pd.notna(row[first]) and pd.notna(row[second])
    }


def _build_profile(code: str, data: dict[str, Any]) -> FundProfile:
    name = data.get("基金名称") or data.get("基金简称")
    if not _optional_text(name):
        raise FundDataError(f"基金 {code} 的名称缺失，无法验证身份。")
    managers = re.split(r"[,，、/\s]+", str(data.get("基金经理", "")).strip())
    return FundProfile(
        code=code,
        name=str(name).strip(),
        full_name=_optional_text(data.get("基金全称")),
        category=_optional_text(data.get("基金类型")),
        investment_type=_optional_text(data.get("投资类型")),
        manager_names=[item for item in managers if item],
        company=_optional_text(data.get("基金公司") or data.get("基金管理人")),
        custodian=_optional_text(data.get("托管银行") or data.get("基金托管人")),
        inception_date=_optional_date(data.get("成立时间") or data.get("成立日期")),
        scale_text=_optional_text(data.get("最新规模") or data.get("份额规模")),
        benchmark=_optional_text(data.get("业绩比较基准")),
        strategy=_optional_text(data.get("投资策略")),
        objective=_optional_text(data.get("投资目标")),
        management_fee=_optional_text(data.get("管理费")),
        custody_fee=_optional_text(data.get("托管费")),
        max_purchase_fee=_optional_text(data.get("最高申购费")),
        max_redemption_fee=_optional_text(data.get("最高赎回费")),
    )


def _latest_available_year(function: Callable[..., pd.DataFrame], code: str, cutoff: date):
    for year in range(cutoff.year, cutoff.year - 3, -1):
        try:
            frame = function(symbol=code, date=str(year))
        except (KeyError, ValueError, TypeError):
            continue
        if frame is not None and not frame.empty:
            return frame
    return pd.DataFrame()


def _holding_source(endpoint: str, retrieved_at: datetime, holdings: list[FundHolding]):
    periods = [item.disclosure_period for item in holdings if item.disclosure_period]
    period_dates = [_period_end_date(item, retrieved_at.year) for item in periods]
    latest_period = (
        max(periods, key=lambda item: _period_end_date(item, retrieved_at.year))
        if periods
        else None
    )
    return SourceRecord(
        name="天天基金/东方财富（经 AKShare）",
        endpoint=endpoint,
        url=AKSHARE_FUND_DOCS,
        retrieved_at=retrieved_at,
        as_of=max(period_dates) if period_dates else None,
        note=(f"最新返回披露期：{latest_period}；定期报告持仓，不代表实时持仓。" if latest_period else "定期报告持仓。"),
    )


def _optional_text(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value) or value == "":
        return None
    try:
        number = float(str(value).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _non_negative_float(value: Any) -> Optional[float]:
    number = _optional_float(value)
    return number if number is not None and number >= 0 else None


def _optional_date(value: Any) -> Optional[date]:
    if value is None or pd.isna(value) or value == "":
        return None
    converted = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(converted) else converted.date()


def _period_end_date(value: Any, fallback_year: int) -> date:
    text = _optional_text(value) or ""
    match = re.search(r"(\d{4})年([1-4])季度", text)
    if not match:
        return date(fallback_year, 12, 31)
    year, quarter = int(match.group(1)), int(match.group(2))
    month_day = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    month, day = month_day[quarter]
    return date(year, month, day)
