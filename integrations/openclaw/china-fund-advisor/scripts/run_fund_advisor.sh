#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "错误：用法 run_fund_advisor.sh <六位基金代码> [--predict]。" >&2
  exit 2
fi

fund_code="$1"
mode="${2:-}"
if [[ ! "$fund_code" =~ ^[0-9]{6}$ ]]; then
  echo "错误：中国公募基金代码必须是六位数字，例如 110022。" >&2
  exit 2
fi
if [[ -n "$mode" && "$mode" != "--predict" ]]; then
  echo "错误：第二个参数只允许 --predict。" >&2
  exit 2
fi

project_root="${CHINA_FUND_ADVISOR_ROOT:-}"
if [[ -z "$project_root" || "$project_root" != /* ]]; then
  echo "错误：未配置有效的 CHINA_FUND_ADVISOR_ROOT 绝对路径。" >&2
  exit 3
fi

python_bin="$project_root/.venv/bin/python"
if [[ ! -x "$python_bin" || ! -f "$project_root/tradingagents/funds/cli.py" ]]; then
  echo "错误：找不到基金顾问 Python 环境或基金 CLI。" >&2
  exit 4
fi

cd -- "$project_root"
export PYTHONDONTWRITEBYTECODE=1

# Reuse the OpenClaw service secret without duplicating it into the repository.
if [[ -z "${DEEPSEEK_API_KEY:-}" && -n "${OPENCLAW_DEEPSEEK_API_KEY:-}" ]]; then
  export DEEPSEEK_API_KEY="$OPENCLAW_DEEPSEEK_API_KEY"
fi
export DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"

if [[ "$mode" == "--predict" ]]; then
  exec timeout --signal=TERM --kill-after=15s 600 \
    "$python_bin" -m tradingagents.funds.cli "$fund_code" --predict
fi

exec timeout --signal=TERM --kill-after=15s 300 \
  "$python_bin" -m tradingagents.funds.cli "$fund_code"
