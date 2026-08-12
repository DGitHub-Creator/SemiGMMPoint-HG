# ODPT NaN 崩溃与协议泄漏修复报告（SemiGMMPoint）

日期：2026-08-10

## 1. 问题

官方 10% ODPT 训练（`run_10.sh`，100 epochs，2×RTX 4090，DDP）在 **Epoch 30 无监督轮第 14/48 步**崩溃（rank 1）：

```
ValueError: The value argument must be within the support
```

（`GMMHead.compute_log_prob` → `multivariate_normal_diag.log_prob`。torch 1.9.0+cu111 中
`torch/distributions/constraints.py` 的 `_Real.check = value == value`，故该异常等价于输入含 NaN。）

同时发现协议泄漏：训练期间每 epoch 用 unlabeled pool 中的 `Area_1_conferenceRoom_3/7` 的
GT 做 val 并选择 best 检查点（`_ckpt_best.pth`，epoch 19）——ODPT 协议不允许。

## 2. 根因

NaN 传播链（单元测试逐一证实）：

1. `ChromaticAutoContrast`（`point_transform_cpu.py:192`）：`scale = 255/(hi-lo)`，常量颜色通道
   hi==lo → scale=inf → 颜色 NaN（p=0.2 随机触发）。测试：30000 点常量通道输入 → 10000 NaN。
2. 编码器 ReLU 坍缩产生全零特征行；`GudiePointContrastLoss` 原实现
   `feat / torch.norm(feat, p=2, dim=1, keepdim=True)` 对零行除零 → **loss=NaN**（测试：1 个零行 → 64 NaN）。
3. 首段 loss 无 NaN 检查 → NaN loss → NaN 梯度 → DDP all-reduce 后参数全 NaN → 下一 forward
   `_c = l2_normalize(feat_norm(base_feature))` 变为 NaN → `log_prob` 触发 ValueError。

GMMHead 自身（means/diagonal/queue，epoch 29 检查全 finite；diagonal min=0.156；cal_id_loss
分母安全；`update_GMM` 走 eps-clamped `l2_normalize`）不是 NaN 源，只是受害方。

## 3. 修复清单

| 文件 | 修改 |
|---|---|
| `openpoints/transforms/point_transform_cpu.py` | ChromaticAutoContrast 常量通道（hi==lo）跳过 scale（保持原值） |
| `openpoints/loss/gudie_point_contrast_loss.py` | `/torch.norm(...)` → `F.normalize(p=2,dim=1)`（eps-clamped，零行安全）；删除 exit(0) 调试；接入 assert_finite；新增伪标签诊断（pl_stats/nfinite）；空类别防护 |
| `openpoints/models/DecodeHead/GMMHead.py` | `_c`/means/diagonal 加入 assert_finite；空队列组件跳过 EM 更新；EM 更新后 DDP all-reduce 同步 means/diagonal |
| `openpoints/utils/finite_check.py` | 新增：assert_finite/assert_grads_finite/assert_params_finite/configure_finite_check；首个非有限张量打印完整上下文（shape/dtype/epoch/iter/rank/NaN/Inf/min/max/mean/std）后 destroy_process_group+os._exit(1) |
| `examples/segmentation/semi_gmmpoint_main.py` | MMCV 公告警告定向过滤；NCCL `torch.cuda.set_device` 前置 + rank/gpu/pid 日志；`--resume_from` 全状态恢复（model+optimizer+scheduler+epoch+GMM）；`--disable_validation`（跳过 validate 与 best 选择，日志输出 `validation=DISABLED (checkpoint_selection=FINAL_EPOCH_ONLY, test_area_evaluation=DID_NOT_RUN)`）；无监督轮 Acc→N/A + 伪标签指标（ratio/coverage/finite ratio）+ 空混淆矩阵防护；sup/unsup 每步 loss 与每 step 梯度/参数 finite 检查；tqdm rank0-only + ascii |
| `cfgs/odpt/default.yaml` | `save_freq: 5`（每 5 epoch 里程碑 `_E%.3d.pth`）、`num_debug_gmm`、`disable_validation` |
| `cfgs/odpt/semi_gmm_odpt_split{10,20}.yaml` | `is_load_gmm: False`、`pretrain_path: null`（死配置清理） |
| `scripts/odpt/*` | 入口固定 `cd $REPO_ROOT`（修复 CWD 相对路径把实验目录写进 scripts/odpt 的 bug）；TF_CPP_MIN_LOG_LEVEL=3、LANG/LC_ALL/PYTHONIOENCODING/PYTHONUNBUFFERED；`RESUME_FROM` 透传 `--resume_from`；CLI 强制 `--disable_validation True --num_debug_gmm True` |
| 崩溃运行目录 | `_ckpt_best.pth` 已改名 `_ckpt_best_INVALID_PROTOCOL_DEBUG_ONLY.pth`（协议无效，仅调试） |
| `.gitignore` | 新增（__pycache__/build/*.egg-info/*.so 等） |

## 4. 验证

单元测试（全部通过）：
- ChromaticAutoContrast 常量通道：200 次 × 30000 点，0 NaN。
- GudiePointContrastLoss 零行输入：loss 有限；伪标签路径（all-ignore→argmax）正常；
  pl_stats/nfinite 输出正确。
- GMMHead.compute_log_prob 注入 1000 NaN：assert_finite 先于 ValueError 触发并退出（打印
  tensor/NaN count 上下文）。

两卡 DDP 训练验证（`experiments/odpt/odpt/...`）：
- 冒烟（2 epochs）：TRAIN_OK；日志含协议行 `labeled 2 / unlabeled 13 (incl. val scenes ...)`、
  `Area_3 is NOT used`、`validation=DISABLED`；无 MMCV 警告、无 NaN；NCCL rank/gpu/pid 行；
  `_E5.pth` 里程碑生成；`--resume_from` 从 epoch2 续跑成功（start_epoch 正确）。
- **崩溃点复现验证**（核心）：从崩溃现场 epoch-29 `_ckpt_latest.pth` 恢复（start_epoch=30，
  保留 optimizer/scheduler/epoch/GMM 状态），两卡跑到 **epoch 35**：
  - **Epoch 30（原崩溃点）**：unsup loss 1.5255，pseudo-label ratio 0.8318，coverage 1.0000，
    finite-feature ratio 1.0000，Acc N/A —— **通过，无 NaN 无崩溃**。
  - **Epoch 35**（第二个无监督周期）：unsup loss 1.5762 —— 通过。
  - 全程 0 次 FINITE_CHECK/NONFINITE/ValueError；日志中 "INF" 命中全部为日志前缀 "INFO"。
  - `_E30.pth`/`_E35.pth` 里程碑 + ckpt_latest（epoch 35）落盘。

## 5. 遗留说明

- 正式协议只认 `experiments/odpt/10pct/checkpoints/official/final.pth`（epoch 100 时由
  `--odpt_final_ckpt` 写出，随后 `run_10_eval.sh` 在 3 个 Area_3 场景评测）。本次验证运行
  使用 `RUN_ID=verify`，未污染 official 输出。
- 验证运行不带 `--num_debug_gmm True` 时，诊断完全关闭（零开销）；带 `--num_debug_gmm True`
  时在 NaN 出现时立即报告并终止（不再以 ValueError 崩溃）。
- 两个无监督周期跑在"全零采样间隔"下性能正常（48 iters/epoch ~2.3 min），未观察到
  无监督轮速度异常。
- 警告治理仅过滤 MMCV v2.0 公告行；未改动任何依赖版本。

---

# 追加报告：Epoch 20 新崩溃（guide loss 维度边界）修复与验证（2026-08-10 下午）

## 6. 新崩溃现象

后续正式 10% 训练（`run_10_train.sh`，100 epochs）在 **Epoch 20 无监督轮**再次崩溃（rank 1），
同一 epoch 内每批一个 crash 点、共四个 crash 点（前三个 `_odpt_unsup_epoch20_batchXX` 目录），
均在 `gudie_point_contrast_loss.py` 旧实现（崩溃时版本）的 **行 132**：

```
File "openpoints/loss/gudie_point_contrast_loss.py", line 132, in forward
    loss = self.criterion(out, labels)
File ".../torch/nn/modules/loss.py", line 1128, in forward
    return F.cross_entropy(input, target, weight=self.weight,
File ".../functional.py", line 2844, in cross_entropy
    return torch._C._nn.cross_entropy_loss(input, target, weight, ...)
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

## 7. 根因（与第 2 节不同：本崩溃与 NaN 无关）

崩溃点在 guide 对比项 `out` 喂给 `nn.CrossEntropyLoss` 之前。旧实现在计算 logits 后有一个
**裸 `squeeze()`**：

- 当某 rank 当前批的伪标签类别数 **K=1** 时，`out` 形状为 `[1, 1]`（一行一列，类维即类数）；
- `out.squeeze()` 把 `[1, 1]` 压成 **`[]`（标量）**，`CrossEntropyLoss` 期望输入 `[N, C]`
  （`input.dim() >= 2`），标量直接触发 `IndexError: Dimension out of range ... got 1`；
- K>1 时 squeeze 无害（`[K,K]→[K,K]`，因为 `[K,K]` 两个维度都 >1），K=0 时走 `continue` 分支，
  所以只有 **K=1** 会炸 —— 与"Epoch 20 才出现"一致：此时模型刚完成 E15/E20 两个无监督轮，
  伪标签第一次出现单类别主导的批次（rank 0 与 rank 1 每批看到的伪标签类别数不同，
  DDP 下只有 rank 1 的某一批恰好 K=1）。

两卡 DDP 下每 rank 独立采样、loss 逐 rank 计算（DDP 只 all-reduce 梯度），所以只有 rank 1
崩溃、rank 0 继续跑 —— 与现场一致。

## 8. 修复（`openpoints/loss/gudie_point_contrast_loss.py` 重写 guide/NCE 路径）

| 位置（当前文件行号） | 修改 |
|---|---|
| 行 14-18（`NCESoftmaxLoss`） | 删除裸 squeeze；仅允许 `reshape(-1, C)` 展平批维，类维恒保留 |
| 行 62-104（`_check_labels` / `_check_ce`） | 新增输入契约检查：标签必须在 `[0, C)` 或 ignore；`out` 必须 `ndim==2`（K=1 时 `[1,1]` 合法、squeeze 后标量报错并给出明确信息）、形状/类别范围/有限性检查，违规即带完整上下文报错 |
| 行 193-199（K=0） | `count.numel()==0`（无有效类别）或 `min(count)<3`（每类点数不足，`min_iter<1`）→ 跳过该批项，**K=0 贡献可微零**（`loss = x.sum()*0`），保持计算图连通、DDP 无 unused 参数 |
| 行 202-220（guide 循环） | **无 squeeze**；`out` 恒为 `[K,K]`（`K>=1`），labels `[K]`；`_check_ce` 每次迭代校验；K=1 时 out 保持 `[1,1]`（注释行 216 明确说明） |
| 行 106-108 | 入口校验 3D features/logits、批维一致、`seg_pred` NaN 拒绝 |
| 行 236-252 | 诊断：`pl_stats`（coverage / accepted_ratio / accepted_guide_points / guide_ce_skipped / nce_skipped / used_pseudo）、`nfinite`；每次 forward 重置 |

**K 处理规则（本次修复核心）**：
- `K=0`（无类别/类别点数 <3）：贡献可微零，不建图 —— DDP backward 不需要它，参数梯度由
  同批其他项提供；
- `K=1`（单类别）：`out` 保持 `[1,1]`、`labels=[0]`，`CrossEntropy([1,1],[0])` 是合法一类别
  分类（P(class0)=1），有梯度有贡献；
- `K>1`：正常 `[K,K]` 多类对比。

## 9. 验证（全部通过）

**9.1 单测 `tools/test_odpt_guide_loss_edges.py`：64/64 通过**，覆盖
K=0（<3 点/类）、K=1（`[1,1]` 不炸、有梯度）、K=2、K=4096（npos 上限）、全 255 伪标签、
单类、非连续标签、B=1 与 B=4、非法标签（越界报错）、NaN logits（入口拒绝）、形状校验。

**9.2 两卡 DDP 边界测试 `tools/test_odpt_guide_loss_ddp_edges.py`：通过**
（真实 NCCL DDP + backward + step）：
- Phase A：rank0 K=1（40 guide pts）vs rank1 K=2（80）—— 不同 rank 不同 K 不挂起；
- Phase B：rank0 K=0（4 项全跳过）vs rank1 K=2 —— K=0 可微零路径正常；
- 全程梯度/参数 finite，无 NCCL 卡死。

**9.3 从失败现场恢复（核心）**：从崩溃运行 `...105827-...` 的 `_ckpt_latest.pth`
（epoch 19，保留 optimizer/scheduler/epoch 全状态）恢复，`RUN_ID=resume_debug EPOCHS=35`
跑到 **epoch 35**：Epoch 20（原崩溃点）通过，E25/30/35 三个无监督轮全部通过，
`pseudo_label_ratio 0.87-0.94`、`coverage 1.0000`、`finite-feature ratio 1.0000`、
guide CE skipped 1-7 项/epoch，0 次 FINITE_CHECK/NONFINITE/ValueError，exit=0。

**9.4 从头重跑 35 epoch（最终验证）**：`RUN_ID=verify35 EPOCHS=35`（正式 100 epoch 未启动，
official 目录保持干净）：
- E1 起全程无崩溃无挂起，E5/10/15/20/25/30/35 全部通过；
- E20（原崩溃点）unsup loss 1.7197、ratio 0.79、coverage 1.0 —— 通过；
- 无资源泄漏：`resource_tracker` / `leaked semaphore` 日志 0 次；
- 协议行齐全：`PROTOCOL_CHECK_PASSED`、`labeled 2 / unlabeled 13`、`validation=DISABLED`、
  `Area_3 is NOT used`、`Training from scratch`。

**9.5 过程修复**：`main_cleanup` 提为模块级函数（mp.spawn 可 pickle）；unsup epoch 汇总行
`pseudo_label_ratio/coverage` 从最后一次 forward 捕获（不再显示 nan）。

## 10. Checklist

- [x] 新崩溃（Epoch 20 行 132 IndexError）根因定位：guide logits `[1,1]` 裸 squeeze → 标量；
- [x] 修复：删除裸 squeeze，`_check_ce` 强制 `ndim==2`，K=0/1/>1 全部有定义且有测试；
- [x] 复现验证：从 epoch 19 ckpt 恢复跑到 35，E20/25/30/35 全过；
- [x] 从头验证：35 epoch 全过（未启动正式 100 epoch）；
- [x] 协议合规：val 完全禁用、仅 2 labeled + 13 unlabeled、无 Area_3、无 best-ckpt 污染；
- [x] 无 resource_tracker 泄漏；`git diff --stat`：12 个改动文件（`git status` 见下）。

## 11. 变更文件（`git status --short`）

```
M examples/segmentation/gmm_main.py
M examples/segmentation/pre_gmm_main.py
M examples/segmentation/semi_gmmpoint_main.py
M openpoints/dataset/__init__.py
M openpoints/dataset/s3dis/__init__.py
M openpoints/loss/gudie_point_contrast_loss.py
M openpoints/models/DecodeHead/GMMHead.py
M openpoints/models/DecodeHead/__init__.py
M openpoints/models/DecodeHead/utils/__init__.py
M openpoints/models/DecodeHead/utils/wapper.py
M openpoints/models/segmentation/base_seg.py
M openpoints/transforms/point_transform_cpu.py
?? .gitignore
?? ODPT_NAN_AND_PROTOCOL_FIX_REPORT.md
?? cfgs/odpt/  experiments/  openpoints/dataset/s3dis/odpt.py
?? openpoints/utils/finite_check.py  scripts/  tools/
```

`git diff --stat`：12 files changed, 539 insertions(+), 178 deletions(-)（+ 未跟踪新增）。
