#!/usr/bin/env bash
# =============================================================================
# chain_opt —— 优化器层扩展实验臂（SWA / Lookahead / 组合 / no-amp 探针）
# -----------------------------------------------------------------------------
# 用法：
#   bash scripts/chain_opt.sh <TAG> "<额外训练参数>"
#
# 例：
#   bash scripts/chain_opt.sh swa            "--swa"
#   bash scripts/chain_opt.sh lookahead      "--lookahead"
#   bash scripts/chain_opt.sh swa_la         "--swa --lookahead"
#   bash scripts/chain_opt.sh noamp          "--no-amp"          # ★ fp32 贡献探针
#
# 唯一变量 = 优化过程（SAM 已证该层能动；SWA/Lookahead 同族，此前全仓零文件）。
#   - SWA：训练末期对 raw 权重滑动平均，平均权重当提交权重（跨域更稳）。
#   - Lookahead：快权重探索、慢权重收敛（k=5, α=0.5 文献标准值）。
#   - no-amp：关 AMP 走 fp32，探「SAM 包的 +1.55pp 里多少是 fp32 白送」（知识臂）。
#
# ★ 全部只改优化过程，不加参数/不改结构/不碰测试集 ⇒ 合规。
# ★ 默认全关 ⇒ 冠军复现路径零影响（src/swa_lookahead.py + finetune_lora.py 注入）。
# ★ 不自动提交（取最高制无下行风险，但榜分是唯一裁决；交哪份由人定）。
# =============================================================================
set -u

# ---------- CR 闸门（工程约定：.sh 含 CR ⇒ bash 静默崩） ----------
if [ "$(tr -cd '\r' < "$0" | wc -c)" -ne 0 ]; then
  echo "🛑 $0 含 CR ⇒ bash 会先崩" >&2; exit 9
fi

TAG="${1:-}"
EXTRA="${2:-}"
KIND="${3:-sce}"          # ★ 2026-10-09 增：损失类型（默认 sce；GCE 臂传 gce）
if [ -z "$TAG" ]; then
  echo "用法：bash scripts/chain_opt.sh <TAG> \"<额外训练参数>\" [KIND]" >&2
  echo "  例：bash scripts/chain_opt.sh swa --swa" >&2
  echo "      bash scripts/chain_opt.sh gce \"\" gce" >&2
  exit 1
fi

RR="${AIC_ROOT:-/root/autodl-tmp/aic}"
PY=/root/miniconda3/bin/python
LOG=$RR/outputs/chain_opt.log
L2=$RR/outputs/${TAG}_run.log
OUT=$RR/outputs/d_$TAG
SUB=$RR/outputs/submission_d_$TAG
PB=$RR/outputs/cache/probs_test_d_$TAG
READY=$RR/outputs/pulled_ready
CONF=$RR/outputs/dedup_soft_conflicts.json
cd "$RR" || exit 1

export AIC_IMG_SIZE=448
export OMP_NUM_THREADS=8
say(){ echo "[chain_opt($TAG) $(date '+%F %T')] $*" | tee -a "$LOG"; }

LOCK=$RR/outputs/.chainopt_${TAG}.lock
if [ -f "$LOCK" ]; then
  OP=$(cat "$LOCK" 2>/dev/null || echo -)
  if [ "$OP" != "-" ] && kill -0 "$OP" 2>/dev/null; then
    say "🛑 已有同 TAG 实验在跑（pid=$OP）⇒ 退出"; exit 0
  fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

say "===== chain_opt($TAG) 启动 pid=$$ 额外参数=[$EXTRA] ====="

# ---------- ① 装置与原料自证 ----------
TORCH_OK=$($PY -c "import torch;print(int(torch.cuda.is_available()))" 2>&1 | tail -1)
say "  cuda.is_available=$TORCH_OK"
[ "$TORCH_OK" = "1" ] || { say "🛑 cuda 不可用 ⇒ 中止"; exit 3; }
[ -f "$CONF" ] || { say "🛑 缺清单 $CONF ⇒ 中止"; exit 3; }
say "  清单：$CONF（$(stat -c%s "$CONF") B）"
say "  磁盘可用：$(df -h /root/autodl-tmp | tail -1 | awk '{print $4}')"

# ---------- ② 源码与损失自证（优化器层改动不能偷换目标函数） ----------
say "  ---- 源码自证 ----"
say "  src/finetune_lora.py sha256 = $(sha256sum "$RR/src/finetune_lora.py" | awk '{print $1}')"
say "  src/swa_lookahead.py sha256 = $(sha256sum "$RR/src/swa_lookahead.py" | awk '{print $1}')"
$PY - <<'PYEOF' 2>&1 | sed 's/^/    /' | tee -a "$LOG"
import sys, torch, torch.nn.functional as F
sys.path.insert(0, 'src')
import importlib, noise as NZ
importlib.reload(NZ)
r = NZ.soft_selfcheck()
for k, v in r.items():
    print(f"[自证] {k} = {v}")
assert r["sce_soft_vs_sce"] < 1e-6, "软目标 SCE 与硬标签不逐位一致 ⇒ 中止"
assert r["ce_soft_vs_ce"] < 1e-6, "ce 软目标不一致 ⇒ 中止"
assert r["soft_vs_hard_differs"] > 1e-3, "软目标与硬目标无差异 ⇒ 实现退化 ⇒ 中止"
import config as C
assert C.LOSS_KIND == "sce", f"默认 kind 不是 sce ⇒ 中止"
assert abs(C.SCE_BETA - 0.05) < 1e-12, "SCE_BETA 变了 ⇒ 中止"
assert C.SEED == 0, "SEED 变了 ⇒ 中止"
print("[自证] ✅ 软目标与硬标签 one-hot 逐位一致；config: sce / SCE_BETA=0.05 / SEED=0")
PYEOF
[ ${PIPESTATUS[0]} -eq 0 ] || { say "🛑 源码/软目标自证失败 ⇒ 中止"; exit 4; }

# ---------- ③ 清单自证 ----------
say "  ---- 冲突组清单自证 ----"
$PY - <<PYEOF 2>&1 | sed 's/^/    /' | tee -a "$LOG"
import json, collections
rec = json.load(open("$CONF", encoding="utf-8"))
gs = rec["groups"]
for g in gs:
    assert abs(sum(g["soft_target"].values()) - 1.0) < 1e-4, "软目标未归一"
    assert len(g["labels"]) == len(set(g["labels"])), "组内标签未去重"
tot_w = sum(g["weight_each"] * len(g["members"]) for g in gs)
assert abs(tot_w - len(gs)) < 1e-3, "组总权重 ≠ 1 ⇒ 中止"
print(f"[清单] 组数={rec['n_groups']} 组总权重合计={tot_w:.1f}（应=组数）")
print("[清单] ✅ 组总权重=1 且软目标归一")
PYEOF
[ ${PIPESTATUS[0]} -eq 0 ] || { say "🛑 清单自证失败 ⇒ 中止"; exit 5; }

# ---------- ④ 训练（CHAMP_BASE 与冠军逐字一致，仅追加优化器层开关） ----------
CHAMP_BASE="--epochs 12 --batch 96 --workers 8 --lr 2e-4 --head-lr-mult 5 \
--targets q,k,v,out,fc1,fc2 --dropout 0.1 --mixup 0.2 \
--ema-decay 0.999 --ema-start 1 --unfreeze-last 4 \
--preproc --track-lowsim --select-by lowsim_a --log-every 200"

FREE=$(df -BG --output=avail "$RR" | tail -1 | tr -dc '0-9')
say "  开跑前可用磁盘 = ${FREE}G（本臂峰值约 2.1G）"
[ "$FREE" -lt 4 ] && { say "🛑 磁盘 ${FREE}G < 4G ⇒ 中止"; exit 7; }
mkdir -p "$OUT"

say "===== [$TAG] 训练开始 ====="
say "  参数：$CHAMP_BASE --hidden 1024 --kind $KIND --r 16 --soft-conflicts $CONF $EXTRA"
AIC_ROOT=$RR \
AIC_CLIP_DIR=$RR/models/clip-vit-base-patch32 \
AIC_IMG_SIZE=448 \
AIC_EXTRACT_WORKERS=8 \
PYTHONPATH=$RR/src \
$PY -u scripts/arm_runner.py -- $CHAMP_BASE \
    --hidden 1024 --kind "$KIND" \
    --soft-conflicts "$CONF" \
    --r 16 --ckpt "$OUT/lora.pt" --out-dir "$OUT" \
    $EXTRA >> "$L2" 2>&1
RC=$?
say "[$TAG] 训练 rc=$RC"
if [ "$RC" -ne 0 ]; then
  say "  STOP 训练失败；末 20 行："; tail -n 20 "$L2" | sed 's/^/    /' | tee -a "$LOG"; exit 8
fi

# ---------- ⑤ 生效自证（硬） ----------
say "  ---- 生效自证 ----"
grep -aE '\[train\] kind=|\[model\] 可训练|\[unfreeze\]|\[soft\]|\[data\]|\[preproc\]|\[val\]' "$L2" | head -14 \
  | sed 's/^/    /' | tee -a "$LOG"
say "  最优：$(grep -aE '最优 epoch' "$L2" | tail -1 | sed 's/^ *//')"
{
  # ★ 2026-10-09：kind 自证放宽到 sce|gce（GCE 臂需要），但仍拦"被悄悄改成 ce"
  KIND_LINE=$(grep -aoE '\[train\] kind=[a-z_]+' "$L2" | tail -1)
  say "  kind 自证：${KIND_LINE:-（未打印）}"
  echo "$KIND_LINE" | grep -qE 'kind=(sce|gce)$' \
    || { say "  🛑 kind 异常（应为 sce 或 gce）⇒ 中止"; exit 9; }
  grep -aqE '可训练 32,301,295 / 155,225,584|可训练 32301295' "$L2" \
    || { say "  🛑 参数量 ≠ 冠军值 ⇒ 非干净对照，中止"; exit 9; }
  grep -aqE '解冻视觉塔最后 4/12 层|解冻 last 4/12' "$L2" \
    || { say "  🛑 未见解冻 4/12 ⇒ 与冠军不一致，中止"; exit 9; }
  HIT=$(grep -aoE '\[soft\] 冲突组=[0-9]+ 训练集命中=[0-9]+' "$L2" | tail -1)
  say "  软目标命中：${HIT:-（未打印）}"
  echo "$HIT" | grep -qE '训练集命中=[1-9]' \
    || { say "  🛑 软目标命中数为 0 ⇒ 等同冠军臂，白跑 ⇒ 中止"; exit 9; }
  # ★ 优化器层开关确认打印（防止「以为开了其实没开」）
  grep -aqE '\[swa\] 已启用|\[lookahead\] 已启用' "$L2" \
    || say "  ⚠ 未检测到 [swa]/[lookahead] 启用日志（EXTRA=[$EXTRA]，可能是 no-amp 探针，可忽略）"
  # ★ 2026-10-08 晚：sel 臂专用生效检查（仅当 EXTRA 带 sel-frac 时才启用）
  #   理由：若 [sel] 未生效 ⇒ 本臂等同冠军臂 ⇒ 白跑 2h，必须早拦。
  #   用宽松匹配（任何 [sel] 行）以防标点漂移误杀好臂。
  if echo "$EXTRA" | grep -q 'sel-frac'; then
    grep -aq '\[sel\]' "$L2" \
      || { say "  🛑 sel 未生效（日志中无任何 [sel] 行）⇒ 等同冠军臂 ⇒ 中止"; exit 9; }
    _SEL_N=$(grep -ac '\[sel\]' "$L2" || true)
    say "  sel 生效确认：[sel] 行 ${_SEL_N} 条 ✓"
  fi
  # ★ 2026-10-09：ASAM 变体专用生效检查（仅当 EXTRA 带 sam-variant asam 时启用）
  if echo "$EXTRA" | grep -q 'sam-variant asam'; then
    grep -aq '\[sam\] ★ 变体 = ASAM' "$L2" \
      || { say "  🛑 ASAM 变体未生效（未见 [sam] ★ 变体 = ASAM）⇒ 等同原版 SAM 臂 ⇒ 中止"; exit 9; }
    say "  ASAM 生效确认：变体行 ✓"
  fi
  # ★ 2026-10-09：LLRD 生效检查（仅当 EXTRA 带 llrd-decay 时启用）
  if echo "$EXTRA" | grep -q 'llrd-decay'; then
    grep -aq '\[opt\] ★ LLRD 启用' "$L2" \
      || { say "  🛑 LLRD 未生效（未见 [opt] ★ LLRD 启用）⇒ 等同平铺单档臂 ⇒ 中止"; exit 9; }
    say "  LLRD 生效确认：$(grep -a '\[opt\] ★ LLRD 启用' "$L2" | tail -1 | sed 's/^ *//')"
  fi
  # ★ 2026-10-09：degrade 生效检查（仅当 EXTRA 带 degrade 时启用）
  if echo "$EXTRA" | grep -q 'degrade'; then
    grep -aq '画质退化增强已启用' "$L2" \
      || { say "  🛑 degrade 未生效（未见 [train] ★ 画质退化增强已启用）⇒ 等同无退化臂 ⇒ 中止"; exit 9; }
    say "  degrade 生效确认：$(grep -a '画质退化增强已启用' "$L2" | tail -1 | sed 's/^ *//')"
  fi
  # ★ 2026-10-10：注入位置限制（--lora-last-n）生效检查
  if echo "$EXTRA" | grep -q 'lora-last-n'; then
    grep -aq '注入位置限制' "$L2" \
      || { say "  🛑 注入位置未生效（未见 [lora] ★ 注入位置限制）⇒ 等同全层注入臂 ⇒ 中止"; exit 9; }
    say "  注入位置确认：$(grep -a '注入位置限制' "$L2" | tail -1 | sed 's/^ *//')"
    grep -a '可训练' "$L2" | head -1 | sed 's/^/    /' | tee -a "$LOG"
  fi
  # ★ 2026-10-10：sim_max 带重加权生效检查
  if echo "$EXTRA" | grep -q 'simmax-rew'; then
    grep -aq '\[smrw\] ★ sim_max 重加权已启用' "$L2" \
      || { say "  🛑 sim_max 重加权未生效（未见 [smrw] 启用行）⇒ 等同无重加权臂 ⇒ 中止"; exit 9; }
    say "  smrw 生效确认：$(grep -a '\[smrw\] ★ sim_max 重加权已启用' "$L2" | tail -1 | sed 's/^ *//')"
  fi
  # ★ 2026-10-10：教师蒸馏（KD）生效检查（仅当 EXTRA 带 kd-probs 时启用）
  if echo "$EXTRA" | grep -q 'kd-probs'; then
    grep -aq '\[kd\] ★ 教师蒸馏已启用' "$L2" \
      || { say "  🛑 KD 未生效（未见 [kd] ★ 教师蒸馏已启用）⇒ 等同无教师臂 ⇒ 中止"; exit 9; }
    _KD_LINE=$(grep -a '\[kd\] ★ 教师蒸馏已启用' "$L2" | tail -1 | sed 's/^ *//')
    say "  KD 生效确认：${_KD_LINE}"
    echo "$_KD_LINE" | grep -qE '命中=0' \
      && { say "  🛑 KD 命中=0（路径对齐失败）⇒ 中止"; exit 9; }
  fi
  say "  ✅ 生效自证通过：kind=sce ｜ 参数量一致 ｜ 解冻 4/12 ｜ 软目标已命中"
}

CK=$OUT/lora_best.pt; [ -f "$CK" ] || CK=$OUT/lora.pt
[ -f "$CK" ] || { say "  STOP 无 ckpt"; exit 10; }

# ---------- ⑥ 出包（与冠军口径逐字一致） ----------
say "  出包（15 视图 power t=4 + Sinkhorn）"
$PY -u scripts/predict_tta.py --lora "$CK" --views best --fusion power --fusion-t 4 \
    --out-dir "$RR/outputs/_${TAG}_pack" --save-probs "$PB" --force --workers 16 >> "$L2" 2>&1
[ -f "${PB}.npy" ] || { say "  STOP 出概率失败"; exit 11; }
$PY - <<PYEOF 2>&1 | sed 's/^/    /' | tee -a "$LOG"
import numpy as np
p = np.load("${PB}.npy")
print(f"[自证] probs shape={p.shape} dtype={p.dtype} 行和={p.sum(1).min():.4f}~{p.sum(1).max():.4f}")
assert p.shape == (37444, 750), "形状不符 ⇒ 中止"
print("[自证] ✅ 形状正确")
PYEOF
[ ${PIPESTATUS[0]} -eq 0 ] || { say "🛑 形状自证失败 ⇒ 中止"; exit 12; }

$PY -u scripts/calib_sinkhorn_apply.py --probs "${PB}.npy" \
    --out-dir "$SUB" --target uniform >> "$L2" 2>&1
if [ -f "$SUB/pred_results.zip" ]; then
  say "  ✅ PACKAGE READY（$(stat -c%s "$SUB/pred_results.zip") B）"
  cp -f "$SUB/pred_results.zip" "$READY/submission_opt_${TAG}.zip"
  say "  md5 = $(md5sum "$READY/submission_opt_${TAG}.zip" | awk '{print $1}')"
  say "  分布自检：$($PY scripts/_zip_class_check.py "$SUB/pred_results.zip" 2>&1 | tail -2 | tr '\n' ' ')"
else
  say "  STOP 未出包"; exit 13
fi

rm -f "$OUT/lora.pt" "$OUT"/lora_ep*.pt
say "===== chain_opt($TAG) 结束 ====="
say "  产物：$READY/submission_opt_${TAG}.zip"
say "  ★★ **不自动提交**（取最高制无下行风险；榜分是唯一裁决，交哪份由人定）"
say "  ★ 读法：vs 冠军(75.2564) / vs SAM(76.8107) —— 唯一变量是优化器层开关"
