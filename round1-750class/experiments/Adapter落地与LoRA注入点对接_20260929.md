# Adapter 落地方案：与现有 LoRA/VPT 注入点对接（2026-09-29 22:59）

> **为什么做**：官方规则五.2 原文点名「**Adapter** 等参数高效微调方法」，我们**完全没做**。
> 属"换做法"级别（新增一类参数高效微调范式），与 LoRA 正交，不依赖任何噪声识别（区别于
> 已被否掉的 TrustCLIP/TGP）。
> **前提已核**：LoRA/VPT 的注入点、参数分组、冻结逻辑全部读通（见下方"挂接点"）。

---

## 一、设计（Pfeiffer 风格 bottleneck Adapter）

### 结构
```
Adapter(x) = x + W_up( GELU( W_down(x) ) )
  W_down : 768 → bottleneck   （下采样）
  W_up   : bottleneck → 768   （上采样，零初始化）
```

### 插入点
`vision_model.encoder.layers.N.mlp`（`CLIPMLP`，含 fc1 768→3072 / fc2 3072→768）**之后**。
12 层每层 1 个 Adapter（Pfeiffer 版，先做单点，不做 Houlsby 双点）。

### 关键参数
| 项 | 值 | 理由 |
|---|---|---|
| bottleneck | **64** 起步（可扫 32/64/128） | 768→64→768，参数量最小可表达 |
| 初始化 W_down | 正态 std=1e-2 | 小扰动起步 |
| 初始化 W_up | **零** | ★ 初始时 Adapter(x)=x，模型逐位等于原 CLIP ⇒ **不破坏先验**（Adapter 版的"B=0 自检"） |
| 参数量 | 12 层 × 768 × 64 × 2 ≈ **1.18M** | 与 LoRA r16（~0.9M）同量级 |

### 与 LoRA 的关系（正交，可叠加）
- LoRA 是**并联**低秩旁路（改 attention + MLP 的 Linear）
- Adapter 是**串联**瓶颈层（在 MLP 输出后加残差）
- 两者可独立开、可同时开 ⇒ 便于「Adapter-only / LoRA-only / 两者」三臂对照

---

## 二、挂接点（精确到函数/行，全部读完源码确认）

### 1. 新文件 `src/adapter.py`（**新增文件，不被现有链读，现在就能写**）
仿照 `src/vpt.py` 的写法（VPT 是最贴近的模板）：

```python
ADPT_TAG = "adapter"   # ★ 参数名带这个串；刻意避开 "lora_"（mark_only_lora_trainable 会冻结不含 lora_ 的参数）

class Adapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int):
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.normal_(self.down.weight, std=1e-2)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)   # ★ 零初始化 ⇒ 初始恒等
        nn.init.zeros_(self.up.bias)
    def forward(self, x): return x + self.up(F.gelu(self.down(x)))

class MLPWithAdapter(nn.Module):
    """包装 CLIPMLP：输出 = adapter(mlp(x))。只改 mlp 模块，不动 layer 的 forward。"""
    def __init__(self, orig_mlp, dim, bottleneck):
        self.mlp = orig_mlp
        self.adapter = Adapter(dim, bottleneck)
    def forward(self, hidden_states):
        return self.adapter(self.mlp(hidden_states))

def inject_adapter(model, bottleneck=64, verbose=True) -> dict:
    # 遍历 vision_model.encoder.layers，对每层的 mlp 做 setattr 替换
    # 幂等：已包装则先解包（仿 VPT 的 inject_vpt 幂等逻辑）

def adapter_parameters(model, only_trainable=True) -> list:
    # 按名精确收集 ADPT_TAG 参数（仿 VPT.vpt_parameters，绝不用 [p for p in model.parameters() if requires_grad]）

def report(...) -> dict:
    # 自检 4 项：① 注入层数=12 ② 零初始化等价（开 Adapter 前向 vs 原模型，max|Δ|=0）
    # ③ 梯度流向（adapter 有非零梯度）④ 冻结干净（带梯度的冻结参数=0）
```

### 2. `src/finetune_lora.py` 挂接 5 处（**必须等 chainF/chainG 结束才能改，铁律 2**）
全部仿照现有 VPT 的写法：

| # | 位置 | 改动 |
|---|---|---|
| 1 | import 区（~65 行 `import vpt as VPT` 旁） | `import adapter as ADPT` |
| 2 | CLI（~1699 行 `--vpt` 附近） | 加 `--adapter`（int, 默认 0）/ `--adapter-bottleneck`（默认 64）/ `--adapter-lr-mult`（默认 1.0） |
| 3 | 注入（~728 行 `VPT.inject_vpt` 附近） | `if args.adapter: ADPT.inject_adapter(clip_model, args.adapter_bottleneck)` |
| 4 | 参数分组（~905 行） | `adpt_ps = _dedup(ADPT.adapter_parameters(model), seen)`；`params = lora_ps + vpt_ps + adpt_ps + bb_ps + head_ps + scale_ps` |
| 5 | 学习率组（~969 行 vpt 组旁） | `groups.append({"params": adpt_ps, "lr": args.lr * args.adapter_lr_mult, "name": "adapter"})` |

★ 账目闸门（`_n_grouped != n_train` 拒绝训练）与 EMA 追踪（`_train_ids`）**自动覆盖** Adapter 参数，
  只要它们进了 `params` 列表——这是本项目最看重的"不静默漏更新"保障。

### 3. runner `scripts/run_adapter_arm.sh`（仿 `run_vpt_arm.sh`）
冠军配方 + `--adapter 1 --adapter-bottleneck 64`，其余逐字照抄。
产出 `d_r448adpt64` + `submission_d_r448adpt64_balU`。

---

## 三、正确性自检（本项目纪律，缺一不可）

| 项 | 判据 | 对应 VPT/LoRA 先例 |
|---|---|---|
| 零初始化等价 | 注入后**不开训练**，前向输出与原模型 **max\|Δ\|=0** | LoRA 的 B=0 自检（VPT 做不到这点，Adapter 能做到） |
| 梯度流向 | adapter 参数 grad 非零 | VPT report 第 4 项 |
| 冻结干净 | 带梯度的冻结参数 = 0 | VPT report 第 5 项 |
| 参数账目 | 分组总数 = 可训练总数（现有闸门） | `_n_grouped != n_train` 闸门 |
| 接线生效 | 训练日志出现 `[adapter] ★ 注入 12 层` | `[pool] ★ 启用非默认池化` 同理 |

---

## 四、成本与排期

| 项 | 估 |
|---|---|
| 训练 12 轮 | ≈ 95min（Adapter 串联只加少量前向开销，与冠军 LoRA 相当） |
| 出包 15 视图 | ≈ 18min |
| **总** | **≈ 2h ≈ ¥4.6** |
| 排期 | chainG（G-1/G-2/G-3）之后；`adapter.py` 现在就能写，`finetune_lora.py` 改动需等链条清空 |

---

## 五、风险与护栏

1. **CLIPMLP.forward 签名**：`forward(hidden_states)` 单参数，极稳定。写 `adapter.py` 时用
   `inspect.signature` 探测（本项目已因 `co_varnames` 穿透不了 `@wrapper` 装饰器踩过坑，
   见铁律 6c），**不凭猜传参**。
2. **与 `--unfreeze-last` 的交互**：若解冻层包含被包装的 mlp，`L.parameters()` 会同时解冻
   Adapter 参数 ⇒ 但分组时 `_dedup`（adpt_ps 先入 seen）会剔掉重叠 ⇒ **不冲突**（与 VPT 同理）。
3. **单变量保证**：除 `--adapter*` 外逐字照抄冠军配方，避免"改了 Adapter 又改了别的"无法归因。
4. **先做 Adapter-only 臂**（`--lora-scale 0`）验证它**单独**能不能打，再考虑与 LoRA 叠加——
   避免一上来就叠加，分不清是谁的功劳。

---

## 六、下一步（等链条清空后执行）

1. 写 `src/adapter.py` + 自检脚本（现在即可，不碰源文件）
2. 等 chainF/chainG 结束 → 改 `finetune_lora.py` 5 处
3. 写 `run_adapter_arm.sh` → 排入下一批链
4. 首臂 `--adapter 1 --adapter-bottleneck 64 --lora-scale 0`（Adapter-only），看低相似列
