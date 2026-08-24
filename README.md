# TradingAgents-CN-Fund-AI V2

面向中国公募基金的证据型 AI 研究助手。它用 AKShare 获取公开数据，用 Python 计算历史风险与
20/60/250 个交易日的历史情景概率，再让多个受约束的 DeepSeek 子 Agent 独立分析、反驳、审查和裁决。

> 这里的“预测”是历史条件情景统计，不是对未来净值或收益的保证。系统不会连接证券账户，
> 不会自动交易，也不会给出买卖时点、金额或仓位指令。

## 能做什么

- 查询六位基金代码对应的基本资料、历史净值、定期披露持仓和行业分布；
- 计算区间收益、年化收益、波动率、最大回撤、历史单日 VaR 和上涨日比例；
- 对 20、60、250 个交易日生成上涨/震荡/下跌的历史情景比例；
- 输出 P10/P50/P90 历史情景收益区间、负收益比例和路径回撤概率；
- 用样本外 Brier Score 对相似情景模型做基线检验，失败时回退到全历史基线；
- 运行六个分析 Agent、多空交叉反驳、证据审查和最终裁决；
- 接入 OpenClaw，识别普通分析和预测意图。

## 当前边界

- 不支持自动交易、保证收益、个性化买卖结论；
- 不完整支持货币基金专用指标、ETF 盘中行情、官方风险等级、同类排名和个人组合相关性；
- ETF、LOF、QDII 能否返回完整数据取决于 AKShare 对应公开接口；
- AI 不能修改 Python 计算的基础概率；Agent 失败时会降级保留确定性结果；
- 公开数据可能延迟、缺失或发生接口变化，使用前应核对基金合同、招募说明书和产品资料概要。

## 快速开始（Linux / 阿里云）

要求：Python 3.11+、Git。以下命令不会安装或升级 Codex CLI。

```bash
git clone https://github.com/ruirui120/TradingAgents-CN-Fund-AI-V2.git
cd TradingAgents-CN-Fund-AI-V2
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env
chmod 600 .env
```

随后只在服务器本地编辑 `.env`，填写自己的 `DEEPSEEK_API_KEY`。不要把密钥粘贴到聊天、
Issue、日志或 Git 提交中。

普通研究报告：

```bash
.venv/bin/python -m tradingagents.funds.cli 110022
```

V2 历史情景预测与多 Agent 审查：

```bash
.venv/bin/python -m tradingagents.funds.cli 110022 --predict
```

不调用 DeepSeek，仅验证数据和 Python 模型：

```bash
.venv/bin/python -m tradingagents.funds.cli 110022 --predict --no-ai
```

保存 Markdown 报告：

```bash
.venv/bin/python -m tradingagents.funds.cli 110022 --predict --output reports/110022.md
```

## 集成 OpenClaw

先在项目根目录执行：

```bash
bash scripts/install_openclaw_skill.sh
```

该脚本只会在目标不存在时复制 Skill，不会覆盖配置、重启服务或开放端口。然后在
`~/.openclaw/openclaw.json` 中启用 `china-fund-advisor`，并设置：

```text
CHINA_FUND_ADVISOR_ROOT=/你的绝对路径/TradingAgents-CN-Fund-AI-V2
```

DeepSeek 密钥应由 OpenClaw 服务环境以 `OPENCLAW_DEEPSEEK_API_KEY` 注入；包装脚本只在子进程中
映射成 `DEEPSEEK_API_KEY`。修改生产配置或重启 OpenClaw 前请先备份并人工确认。

验证：

```bash
openclaw skills info china-fund-advisor
openclaw skills check
```

## 测试

```bash
.venv/bin/python -m unittest discover -s tests/funds -v
.venv/bin/python -m compileall -q tradingagents/funds tradingagents/deepseek.py tests/funds
bash -n integrations/openclaw/china-fund-advisor/scripts/run_fund_advisor.sh
```

单元测试不需要真实 API Key。涉及 AKShare 或 DeepSeek 的联网验证会受到网络、服务和数据接口状态影响。

## 方法与安全设计

详细方法见：

- [V1 数据与风险指标](docs/guides/china-fund-advisor-v1.md)
- [V2 概率预测与多 Agent 审查](docs/guides/china-fund-advisor-v2.md)
- [密钥与安全说明](SECURITY.md)

所有 Agent 声明必须引用证据编号；未知编号、无证据数字、直接交易指令和保证性语言会被程序拒绝。
概率只来自可复现的 Python 历史情景模型，语言模型只能解释和审查。

## 开源与来源

本独立仓库只包含原项目中 Apache License 2.0 覆盖的基金模块、测试、文档和 OpenClaw 集成。
不包含 `app/`、`frontend/`、旧仓库历史、服务器配置或任何真实密钥。

- 上游项目：[hsliuping/TradingAgents-CN](https://github.com/hsliuping/TradingAgents-CN)
- 原始框架：[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents)
- 许可证：[Apache License 2.0](LICENSE)
- 归属说明：[NOTICE](NOTICE)

内容仅供研究参考，不构成投资建议。
