from datetime import date, timedelta
import unittest

import pandas as pd

from tradingagents.funds.providers.akshare import AkshareFundDataSource, FundDataError


class FakeAkshare:
    def fund_individual_basic_info_xq(self, symbol):
        return pd.DataFrame(
            [
                ("基金代码", symbol),
                ("基金名称", "测试成长混合A"),
                ("基金类型", "混合型-偏股"),
                ("基金经理", "张三 李四"),
                ("基金公司", "测试基金管理有限公司"),
                ("成立时间", "2020-01-02"),
                ("最新规模", "20.5亿"),
                ("业绩比较基准", "沪深300指数收益率*60%+中债指数收益率*40%"),
            ],
            columns=["item", "value"],
        )

    def fund_name_em(self):
        raise AssertionError("basic information fallback should not be used")

    def fund_open_fund_info_em(self, symbol, indicator):
        start = date(2024, 1, 1)
        if indicator == "累计净值走势":
            return pd.DataFrame(
                {
                    "净值日期": [start + timedelta(days=index) for index in range(80)],
                    "累计净值": [1 + index * 0.001 for index in range(80)],
                }
            )
        return pd.DataFrame(
            {
                "净值日期": [start + timedelta(days=index) for index in range(80)],
                "单位净值": [1 + index * 0.001 for index in range(80)],
                "日增长率": [0.1] * 80,
            }
        )

    def fund_portfolio_hold_em(self, symbol, date):
        return pd.DataFrame(
            [{"股票代码": "600000", "股票名称": "浦发银行", "占净值比例": 5.5,
              "持仓市值": 1200.0, "季度": "2024年4季度股票投资明细"}]
        )

    def fund_portfolio_bond_hold_em(self, symbol, date):
        return pd.DataFrame(
            [
                {"债券代码": "019547", "债券名称": "国债16", "占净值比例": 2.0,
                 "持仓市值": 400.0, "季度": "2024年4季度债券投资明细"},
                {"债券代码": "019521", "债券名称": "旧国债", "占净值比例": 1.0,
                 "持仓市值": 200.0, "季度": "2024年1季度债券投资明细"},
            ]
        )

    def fund_portfolio_industry_allocation_em(self, symbol, date):
        return pd.DataFrame(
            [{"行业类别": "银行", "占净值比例": 5.5, "市值": 1200.0,
              "截止时间": "2024-12-31"}]
        )


class AkshareFundDataSourceTests(unittest.TestCase):
    def test_fetch_normalizes_public_fund_data(self):
        dataset = AkshareFundDataSource(FakeAkshare()).fetch(
            "000001", as_of=date(2025, 1, 15)
        )

        self.assertEqual(dataset.profile.code, "000001")
        self.assertEqual(dataset.profile.manager_names, ["张三", "李四"])
        self.assertEqual(len(dataset.nav_history), 80)
        self.assertIsNotNone(dataset.nav_history[-1].accumulated_nav)
        self.assertEqual({item.asset_type for item in dataset.holdings}, {"stock", "bond"})
        self.assertEqual(dataset.industry_allocations[0].industry, "银行")
        self.assertTrue(any("定期报告" in warning for warning in dataset.warnings))
        self.assertTrue(all(source.endpoint for source in dataset.sources))
        bond_source = next(
            source for source in dataset.sources
            if source.endpoint == "fund_portfolio_bond_hold_em"
        )
        self.assertIn("2024年4季度", bond_source.note)

    def test_rejects_non_six_digit_code(self):
        with self.assertRaises(FundDataError):
            AkshareFundDataSource(FakeAkshare()).fetch("1A")

    def test_rejects_money_fund_until_correct_yield_metrics_are_supported(self):
        class MoneyFundAkshare(FakeAkshare):
            def fund_individual_basic_info_xq(self, symbol):
                return pd.DataFrame(
                    [("基金代码", symbol), ("基金名称", "测试货币A"), ("基金类型", "货币型")],
                    columns=["item", "value"],
                )

        with self.assertRaisesRegex(FundDataError, "万份收益"):
            AkshareFundDataSource(MoneyFundAkshare()).fetch("000009")


if __name__ == "__main__":
    unittest.main()
