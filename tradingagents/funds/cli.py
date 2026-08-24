"""Command-line entry point for China fund research and V2 forecasting."""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

from .advisor import ChinaFundAdvisor, DeepSeekFundNarrator, render_markdown
from .providers.akshare import AkshareFundDataSource, FundDataError


def main() -> int:
    parser = argparse.ArgumentParser(description="中国公募基金证据型研究助手")
    parser.add_argument("fund_code", help="6 位基金代码，例如 000001")
    parser.add_argument("--as-of", type=date.fromisoformat, help="分析截止日，格式 YYYY-MM-DD")
    parser.add_argument("--no-ai", action="store_true", help="仅输出数据与代码计算指标")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="生成 20/60/250 日历史情景概率，并执行多 Agent 辩论与审查",
    )
    parser.add_argument("--output", type=Path, help="可选 Markdown 输出路径")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    risk_free_rate = _optional_rate(os.getenv("FUND_ANNUAL_RISK_FREE_RATE"))
    narrator = None if args.no_ai or args.predict else DeepSeekFundNarrator()
    advisor = ChinaFundAdvisor(
        data_source=AkshareFundDataSource(),
        narrator=narrator,
        annual_risk_free_rate=risk_free_rate,
    )
    try:
        if args.predict:
            analysis = advisor.predict(
                args.fund_code, as_of=args.as_of, use_ai=not args.no_ai
            )
        else:
            analysis = advisor.analyze(
                args.fund_code, as_of=args.as_of, use_ai=not args.no_ai
            )
    except (FundDataError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"分析失败：{exc}\n")

    report = render_markdown(analysis)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"报告已保存：{args.output.resolve()}")
    else:
        print(report)
    return 0


def _optional_rate(value: str | None):
    if value is None or not value.strip():
        return None
    rate = float(value)
    if rate < -0.1 or rate > 0.5:
        raise ValueError("FUND_ANNUAL_RISK_FREE_RATE 应使用小数，例如 0.015 表示 1.5%。")
    return rate


if __name__ == "__main__":
    raise SystemExit(main())
