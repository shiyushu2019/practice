#!/usr/bin/env bash
# 单次 CF 变体训练 + 指标记录。
# 用法: bash scripts/sweep_one.sh <gpu> <tag> [额外的 train.py 参数...]
# 例:   bash scripts/sweep_one.sh 0 r1_base --freeze-epochs 0 --lr0 0.001 --epochs 100
#
# 说明: 固定 --variant cf --dataset mixed; 输出权重 weights/<tag>.pt;
#       日志 temp/sweep/<tag>.log; 指标追加到 temp/sweep/<round>.metrics.tsv
set -u

GPU="${1:?need gpu}"; TAG="${2:?need tag}"; shift 2
ROUND="${ROUND:-adhoc}"

CELL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CELL"
PY="${PY:-/data1/conda_envs/jssnu005/sys/cell/bin/python}"
OUT="$CELL/temp/sweep"
mkdir -p "$OUT"
LOG="$OUT/$TAG.log"

echo "[$(date +%H:%M:%S)] START tag=$TAG gpu=$GPU args: $*"
"$PY" train.py --variant cf --dataset mixed --device "$GPU" --out "$TAG" "$@" > "$LOG" 2>&1
RC=$?

# 从 train.py 输出的最终 test 指标行中提取 (第二处出现, 即测试集评估)
LINE="$(grep -aE 'mAP@0\.5=' "$LOG" | tail -1)"
M50="$(sed -n 's/.*mAP@0\.5=\([0-9.]*\).*/\1/p' <<<"$LINE")"
M5095="$(sed -n 's/.*mAP@0\.5:0\.95=\([0-9.]*\).*/\1/p' <<<"$LINE")"
P="$(sed -n 's/.*P=\([0-9.]*\).*/\1/p' <<<"$LINE")"
R="$(sed -n 's/.*R=\([0-9.]*\).*/\1/p' <<<"$LINE")"
M50="${M50:-NA}"; M5095="${M5095:-NA}"; P="${P:-NA}"; R="${R:-NA}"

{
  echo -e "tag\tround\tgpu\trc\tmap50\tmap5095\tprec\trecall\targs"
  true
} >/dev/null

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$TAG" "$ROUND" "$GPU" "$RC" "$M50" "$M5095" "$P" "$R" "$*" >> "$OUT/$ROUND.metrics.tsv"

echo "[$(date +%H:%M:%S)] DONE  tag=$TAG rc=$RC map50=$M50 prec=$P recall=$R"
