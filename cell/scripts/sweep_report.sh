#!/usr/bin/env bash
# 汇总某轮 (或全部) 扫描结果, 按 map50 降序显示, 并标注是否达标。
# 用法: bash scripts/sweep_report.sh [round]
#   达标: map50 >= 0.947 且 prec >= 0.897
set -u
CELL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$CELL/temp/sweep"
ROUND="${1:-}"

FILES=( "$OUT"/*.metrics.tsv )
[ -e "${FILES[0]}" ] || { echo "还没有任何结果 ($OUT/*.metrics.tsv)"; exit 0; }

echo "TAG                 ROUND      MAP50   MAP5095 PREC    RECALL  PASS"
# 合并多个 round 文件 (跳过重复表头), 过滤 rc==0 且 map50 为数值
{
  for f in "${FILES[@]}"; do
    [ -n "$ROUND" ] && [[ "$(basename "$f")" != "$ROUND".metrics.tsv ]] && continue
    awk -F'\t' 'NR==1{next} $1!=""{print}' "$f"
  done
} | awk -F'\t' '
  $5 ~ /^[0-9.]+$/ {
    pass = ($5+0 >= 0.947 && $7+0 >= 0.897) ? "YES" : "-"
    printf "%-19s %-9s %-7.4f %-7s %-7.4f %-7.4f %s\n", $1, $2, $5, $6, $7, $8, pass
  }' | sort -k3 -rn
