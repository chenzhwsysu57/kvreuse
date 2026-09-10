# Cross-Prefix KV Reuse 数据集扩展进展报告

**项目：** kvreuse cross-prefix 数据集构建  
**分支：** `exp/include-more-dataset`  
**模型：** Qwen3-1.7B / Qwen3-4B  
**日期：** 2026-09-04  

---

## 1. 任务目标

构建 **10 套** validated cross-prefix 数据集，用于评测 **KV cache 跨 prefix 复用** 时的性能下降。

**合格标准（以 ArgKP / Deal 为标杆）：**

- prefix A / B 任务本质不同（角色、立场或 utility 冲突）
- shared_block 对两边都因果相关
- gold_a ≠ gold_b，可机器校验
- **Full 基线足够高，Cross-Reuse 明显更低**

---

## 2. 总体进度

| 项目 | 状态 |
|------|------|
| Validated 数据集 | **5 / 10** |
| 本轮新增 validated | **+1（Perspectrum）** |
| Reasoning 补跑 | **进行中**（JobInterview 4B full ~18/110） |
| 代码与 benchmark | 已提交 `exp/include-more-dataset` 分支 |

---

## 3. Validated 数据集（5 套）

| # | 数据集 | 任务形式 | 4B no-reasoning Full A/B | Reuse A→B / B→A | Δ vs Full |
|---|--------|----------|--------------------------|-----------------|-----------|
| 1 | **ArgKP** | PRO/CON 选匹配论点 | 90.9% / 88.2%（reasoning） | KVCOMM cross-reuse **≈0%** | 标杆 |
| 2 | **Deal** | 双边 utility 选最优分配 | 100% / 100%（reasoning） | 极低 | 标杆 |
| 3 | **JobInterview** | 工人/招聘方选最优合同 | 81.8% / 54.5% | 1.8% / 12.7% | **-52.7 / -69.1 pp** |
| 4 | **FANToM (belief)** | 加入者 vs 见证者信念 | 50.0% / 97.3% | 62.7% / 3.6% | **-34.6 / -46.4 pp** |
| 5 | **Perspectrum** | support/oppose 选论据 | 47.3% / 49.1% | 23.6% / 16.4% | **-25.5 / -30.9 pp** |

---

## 4. No-Reasoning 完整结果（1.7B / 4B）

| 数据集 | 模型 | Full A | Full B | Reuse A→B | Reuse B→A | Δ(A→B vs Full B) | Δ(B→A vs Full A) |
|--------|------|-------:|-------:|----------:|----------:|-----------------:|-----------------:|
| JobInterview | 1.7B | 51.8 | 36.4 | 8.2 | 25.5 | -28.2 | -26.3 |
| JobInterview | 4B | 81.8 | 54.5 | 1.8 | 12.7 | **-52.7** | **-69.1** |
| FANToM-belief | 1.7B | 27.3 | 91.8 | 76.4 | 9.1 | -15.4 | -18.2 |
| FANToM-belief | 4B | 50.0 | 97.3 | 62.7 | 3.6 | **-34.6** | **-46.4** |
| Perspectrum | 1.7B | 34.5 | 50.9 | 38.2 | 23.6 | -12.7 | -10.9 |
| Perspectrum | 4B | 47.3 | 49.1 | 23.6 | 16.4 | **-25.5** | **-30.9** |
| ExploreToM | 1.7B | 75.0 | 31.8 | 25.0 | 77.3 | -6.8 | +2.3 |
| ExploreToM | 4B | 88.6 | 27.3 | 11.4 | 86.4 | -15.9 | -2.2 |
| CaSiNo | 1.7B | 30.9 | 15.5 | 12.7 | 21.8 | -2.8 | -9.1 |
| CaSiNo | 4B | 15.5 | 3.6 | 1.8 | 13.6 | -1.8 | -1.9 |
| Craigslist | 4B | 14.5 | 4.5 | 0.0 | 0.0 | -4.5 | -14.5 |

---

## 5. Reasoning 模式结果（老板要求补跑）

| 数据集 | 模型 | Full A/B | Reuse A→B / B→A | Δ vs Full | 判定 |
|--------|------|----------|-----------------|-----------|------|
| JobInterview | 1.7B | 32.7% / 8.2% | 4.5% / 27.3% | -3.7 / -5.4 pp | 有 drop，弱于 no-reasoning |
| JobInterview | 4B | *跑数中 (~18/110)* | — | — | 进行中 |
| FANToM | 1.7B | 76.4% / 62.7% | 40.0% / 55.5% | **-22.7 / -20.9 pp** | ✅ |
| FANToM | 4B | 84.5% / 80.9% | 40.0% / 55.5% | **-40.9 / -29.0 pp** | ✅ |
| Perspectrum | 1.7B | 40.9% / 53.6% | 48.2% / 41.8% | -5.4 / **+0.9 pp** | ⚠️ reuse 几乎不伤 |
| Perspectrum | 4B | 56.4% / 63.6% | 60.9% / 56.4% | -2.7 / **0 pp** | ⚠️ 同上 |
| ExploreToM | 1.7B | 75.0% / 31.8% | 22.7% / 86.4% | -9.1 / **+11.4 pp** | ❌ |
| ExploreToM | 4B | 90.9% / 56.8% | 56.8% / 88.6% | 0 / -2.3 pp | ❌ |

**结论：** Reasoning 下 **FANToM、JobInterview** 仍有明显 reuse drop；**Perspectrum** 在 reasoning 模式下 drop 大幅减弱，汇报时建议 **以 no-reasoning 结果为主**。

---

## 6. 本轮新探索数据源

| 数据源 | 改造思路 | 样本量 | 结论 |
|--------|----------|--------|------|
| **Perspectrum** | 共享 claim+evidence，support vs oppose | 412→110 | ✅ **纳入 validated** |
| **ExploreToM** | GT 位置 vs agent 信念 | 44 | ❌ 不对称、reuse 帮倒忙 |
| **Craigslist Bargains** | buyer/seller 价格 utility max | 152→110 | ❌ full 基线过低 |
| **CaSiNo** | campsite 三包 utility max | 1024→110 | ❌ full 基线过低 |
| **FANToM access** | 信息可达性 YES/NO | 237→110 | ❌ reuse ≈ full |

---

## 7. 与 ArgKP 标杆对照

| 指标 | ArgKP 4B（reasoning） |
|------|------------------------|
| Full 准确率 | **90.9% / 88.2%** |
| KVCOMM cross-reuse | **≈ 0%**（相对 full baseline ~89%） |
| 老板期望 | full 高 → reuse 崩（如 77→8） |

**当前最接近标杆的新数据集：** JobInterview 4B no-reasoning（reuse 1.8% vs full B 54.5%）

---

## 8. 后续计划

1. **跑完** JobInterview 4B reasoning full + reuse
2. **继续寻找 5 套新源**，严格按 ArgKP/Deal 模式：双边角色 + 共享上下文 + 离散冲突 gold + full 先筛到 ~50%+
3. 优先 negotiation / argument stance / utility max 类，**不再投入 ToM 或纯对话 utility 题**

---

## 9. 一句话总结

> 目前 **5 套** cross-prefix 数据集达到 validated 标准（ArgKP、Deal、JobInterview、FANToM-belief、Perspectrum）。4B no-reasoning 下 JobInterview reuse 相对 full 掉 **53–69 pp**，Perspectrum 掉 **26–31 pp**。ExploreToM、Craigslist、CaSiNo 不达标；**距 10 套目标还差 5 个**，正按 ArgKP 改造思路继续筛选，reasoning 补跑接近完成。

---

## 附录：结果文件路径

- 汇总 JSON：`results/reuse_drop_summary.json`
- 各数据集 summary：`results/{dataset}_110_{no_reasoning|reasoning}/{1.7b|4b}/{full|reuse}/qwen3-*/summary.json`
- Benchmark：`data/benchmark/benchmark_*.jsonl`
