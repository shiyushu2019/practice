#!/usr/bin/env bash
# 跨多卡并行调度一批 CF 训练配置。
# 用法: bash scripts/sweep_driver.sh [round]        # round 默认 r1
#      GPUS="0 1" bash scripts/sweep_driver.sh r1   # 指定用哪些卡
# 每张卡串行跑分配给它的若干配置, 卡之间并行。
set -u

ROUND="${1:-r1}"
CELL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CELL"
OUT="$CELL/temp/sweep"; mkdir -p "$OUT"
export ROUND

read -ra GPUS <<< "${GPUS:-0 1 2 3 4 5}"
NG="${#GPUS[@]}"

TAGS=(); ARGS=()
add() { TAGS+=("$1"); ARGS+=("$2"); }

case "$ROUND" in
  # ---- 冒烟测试: 3 轮快速跑通全流程 (验证 CBAM/Focal/DAMS 注入与保存) ----
  smoke)
    add smoke_cf "--freeze-epochs 0 --lr0 0.001 --epochs 3 --patience 100"
    add smoke_freeze "--lr0 0.001 --freeze-epochs 1 --freeze-layers 11 --unfreeze-lr 0.001 --epochs 3 --patience 100"
    ;;

  # ---- 第 1 轮: 基线复现 + 超参粗扫 ----
  r1)
    # 复现 ab_cf: 不冻结, 全程 lr0=0.001, 100 轮
    add r1_base    "--freeze-epochs 0 --lr0 0.001 --epochs 100"
    # 学习率 / 优化器
    add r1_lr2e3   "--freeze-epochs 0 --lr0 0.002 --epochs 100"
    add r1_lr5e4   "--freeze-epochs 0 --lr0 0.0005 --epochs 100"
    add r1_adamw   "--freeze-epochs 0 --optimizer AdamW --lr0 0.001 --epochs 100"
    add r1_sgd     "--freeze-epochs 0 --optimizer SGD --lr0 0.01 --epochs 100"
    # batch
    add r1_b32     "--freeze-epochs 0 --lr0 0.001 --batch 32 --epochs 100"
    add r1_b8      "--freeze-epochs 0 --lr0 0.001 --batch 8 --epochs 100"
    # 数据增强
    add r1_nomosaic "--freeze-epochs 0 --lr0 0.001 --mosaic 0 --epochs 100"
    add r1_cm30     "--freeze-epochs 0 --lr0 0.001 --close-mosaic 30 --epochs 100"
    add r1_norot    "--freeze-epochs 0 --lr0 0.001 --degrees 0 --epochs 100"
    # Focal 超参
    add r1_g1       "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --epochs 100"
    add r1_g3       "--freeze-epochs 0 --lr0 0.001 --gamma 3.0 --epochs 100"
    add r1_a50      "--freeze-epochs 0 --lr0 0.001 --alpha 0.5 --epochs 100"
    # 学习率调度 / 轮数 / CBAM
    add r1_ep200    "--freeze-epochs 0 --lr0 0.001 --epochs 200 --patience 60"
    add r1_lrf05    "--freeze-epochs 0 --lr0 0.001 --lrf 0.05 --epochs 100"
    add r1_red16    "--freeze-epochs 0 --lr0 0.001 --cbam-reduction 16 --epochs 100"
    ;;

  # ---- 第 2 轮: 组合 Round1 赢家 (γ=1.0 / 200轮) + 稳健性 ----
  r2)
    add r2_g1_ep200       "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --epochs 200 --patience 60"
    add r2_g1_ep200_b8    "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --batch 8 --epochs 200 --patience 60"
    add r2_g05_ep200      "--freeze-epochs 0 --lr0 0.001 --gamma 0.5 --epochs 200 --patience 60"
    add r2_g15_ep200      "--freeze-epochs 0 --lr0 0.001 --gamma 1.5 --epochs 200 --patience 60"
    add r2_ep300          "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --epochs 300 --patience 80"
    add r2_g1_ep150       "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --epochs 150 --patience 50"
    add r2_g1_ep200_s1    "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --epochs 200 --patience 60 --seed 1"
    add r2_g1_ep200_s2    "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --epochs 200 --patience 60 --seed 2"
    add r2_g1_ep200_norot "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --degrees 0 --epochs 200 --patience 60"
    add r2_g1_ep200_cm30  "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --close-mosaic 30 --epochs 200 --patience 60"
    add r2_g1_ep200_red16 "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --cbam-reduction 16 --epochs 200 --patience 60"
    add r2_g1_ep200_lrf05 "--freeze-epochs 0 --lr0 0.001 --gamma 1.0 --lrf 0.05 --epochs 200 --patience 60"
    ;;

  *)
    echo "未知 round: $ROUND" >&2; exit 2;;
esac

N=${#TAGS[@]}
echo "ROUND=$ROUND  配置数=$N  GPU 数=$NG  ->  ${GPUS[*]}"

# 初始化指标文件表头
MF="$OUT/$ROUND.metrics.tsv"
printf 'tag\tround\tgpu\trc\tmap50\tmap5095\tprec\trecall\targs\n' > "$MF"

for ((g=0; g<NG; g++)); do
  (
    for ((i=g; i<N; i+=NG)); do
      bash "$CELL/scripts/sweep_one.sh" "${GPUS[$g]}" "${TAGS[$i]}" ${ARGS[$i]}
    done
  ) &
done
wait
echo "ROUND=$ROUND 全部完成, 指标见 $MF"
