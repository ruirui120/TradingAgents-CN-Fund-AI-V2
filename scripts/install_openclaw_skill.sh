#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source_dir="$repo_root/integrations/openclaw/china-fund-advisor"
workspace="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}"
target_dir="$workspace/skills/china-fund-advisor"

if [[ ! -f "$source_dir/SKILL.md" || ! -x "$source_dir/scripts/run_fund_advisor.sh" ]]; then
  echo "错误：OpenClaw Skill 源文件不完整，或运行脚本没有执行权限。" >&2
  exit 2
fi

if [[ -e "$target_dir" ]]; then
  echo "停止：目标已存在，未覆盖：$target_dir" >&2
  echo "请先备份并人工确认后再处理旧版本。" >&2
  exit 3
fi

mkdir -p -- "$workspace/skills"
cp -R -- "$source_dir" "$target_dir"

printf '%s\n' \
  "Skill 已复制到：$target_dir" \
  "项目绝对路径：$repo_root" \
  "下一步：在 OpenClaw 配置中启用 china-fund-advisor，" \
  "并把 CHINA_FUND_ADVISOR_ROOT 设置为上面的项目绝对路径。" \
  "本脚本没有修改 OpenClaw 配置，也没有重启服务。"
