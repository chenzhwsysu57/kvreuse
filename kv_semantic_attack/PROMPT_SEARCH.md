# 独立的 prefix 重述 + 通用提示搜索

这里不使用 attacker、defender、自博弈或 LLM judge。固定任务池、模型与直接
KV reuse 算法，用 Optuna TPE（或随机搜索基线）选择 prompt 构造器。

## 实验定义

追加文本为 `C(prefix_target) + generic_instruction`，位置为：

**原目标 prefix → 复用的 shared-block KV → 重述 + 通用提示 → 原 question**。

- 原 prefix、共享块和 gold 不改动；Full 使用原始输入，不追加重述。
- 压缩器只接收当前方向的 prefix 字符串，不接收 metadata、rule、共享块或 gold。
- 两个方向用同一压缩策略和通用模板，但重述随目标 prefix 改变。
- 使用现有 scatter KV 引擎，不启用 tail recomputation / prefix-KV replay。
- 使用 Qwen 官方 chat template、固定 system、无 reasoning 和原有 boxed 选项判分。

## 第一版搜索空间与限制

内置两种重述方式：

1. `full`：完整保留原 prefix。
2. `compact`：仅去除三种已知的冗余任务开头，保留剩余字节；不截断 token，
   不删除否定、过滤条件、排序方向、tie-break、标签映射或输出约束。
   遇到不认识的开头，回退到完整 prefix。

内置 4 种通用提示、3 种包装方式，共 24 种配置。搜索候选**必须同时包含重述
与通用提示**；不追加、只加通用提示、只加压缩重述是消融对照，不参与最终选优。
内置压缩是保守提取，不是已经验证的语义摘要模型。

通用提示均要求服从后续 question 的输出格式。完整复制 prefix 仍可能重引入
原始 payload 输出要求，不能将它视为准确率上界。保持原文的代价会被如实评测。

### 可选语义压缩库

`--compression-bank` 接收 JSON，顶层 `variants` 是“压缩策略名称 →
原 prefix UTF-8 SHA-256 → 压缩文本”的映射。名称不能使用 `none/full/compact`。
每个策略可代表一个压缩预算或一种摘要方式。

- 哈希针对精确原文，不规范化空格。
- 每个 trial 对所有题目使用同一个策略名称。
- 缺少某个 prefix 的条目时完整保留，并记录 fallback 次数。
- 可提前用人工或 LLM 从 **prefix 文本本身**生成；搜索入口不调用外部 API。
- 不应基于题目答案、rule metadata 或测试分数制作/调整压缩库。程序能限制
  运行时输入，但无法证明外部候选的来源或语义等价性，必须另行审核。
- 长度使用目标 tokenizer 实测记录，不保证外部摘要一定更短。

## 运行入口

使用主 `.venv_kvreuse` 环境，额外依赖见
[requirements-prompt-search.txt](requirements-prompt-search.txt)：`optuna>=4.0,<5`。
运行模块为 `kv_semantic_attack.run_prompt_search`，不要直接执行文件路径。

必填参数：`--input`（合成 paired JSONL）、`--output-dir`（独立运行目录）。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--dry-run` | 关闭 | 校验数据、显示划分及 prompt 示例，不加载模型、不写输出 |
| `--sampler` | `tpe` | `tpe` 或 `random` |
| `--trials` | 16 | 总的成功配置预算，含两个共同 warm-start |
| `--model` | `1.7b` | `0.6b/1.7b/4b/8b`，仅使用本地模型快照 |
| `--batch-size` | 8 | pair 数，每个 pair 评估两个方向 |
| `--max-new-tokens` | 128 | 固定 greedy decode 上限 |
| `--seed` | 20260910 | 同时固定数据划分和搜索随机种子 |
| `--validation-fraction` | 0.2 | 验证集分组占比 |
| `--test-fraction` | 0.2 | 测试集分组占比 |
| `--shortlist-size` | 5 | 从 search 分数排名取前 K 个进入验证集 |
| `--accuracy-tolerance-pp` | 0 | 在验证集上允许用多大准确率损失换取更短文本 |
| `--compression-bank` | 无 | 可选的 prefix-only 语义压缩库 |

GPU 使用 `CUDA_VISIBLE_DEVICES` 选择；每个进程只用可见 `cuda:0`。
第一版不提供跨 trial 多 GPU 并发，可在不同 GPU 上运行不同输出目录的独立实验。

**当前真实评估输入限定为本仓库生成器输出的合成 paired JSONL**，并严格调用
`validate_generated_record()`。不是任意 benchmark JSONL 的通用评估入口；
ArgKP/Deal 等数据需要另行适配判分与数据校验。每种任务分层至少需要三个独立
`shared_data_id`，实际实验应远多于这个最低限度。

## 搜索、选优和对照

1. 按 `shared_data_id` 分组，按组内任务类型集合分层，确定 search/validation/test。
   同资料的方向与不同排版不跨集合。每层至少保留一个验证组和测试组。
2. 评测 search 上的固定消融对照；Full 对每个 split 只计算一次。
3. TPE 最大化 search 的整体 reuse accuracy。前 8 个完成观测使用随机启动；
   因此只跑几次 trial 不能证明贝叶斯搜索有效。
4. search 前 K 名加两个固定候选（完整重述+提示、压缩重述+提示）进入验证集。
5. 在距离**验证候选中的最高准确率**不超过 tolerance 的候选中选追加 token
   均值最小者；同长再按准确率选择。默认 tolerance=0，不主动牺牲准确率。
6. 在读取测试分数之前写入冻结的 selection；随后评估选定方案与预先约定的
   五个消融对照。测试结果不用于回选候选。

固定对照为：direct reuse、generic only、compressed only、compressed+generic、
full+generic；另提供无追加文本的 Full 参考。搜索中其他 full 重述配置也可获选。

公平比较 TPE 与 random 时，使用相同输入、seed、配置预算、模板空间与两个
warm-start，仅换 sampler 和输出目录。配置预算不等于新 GPU 评估次数：
渲染文本相同的配置共享结果缓存，固定对照也有额外开销。需同时比较实际推理
时间/不同评估文件数，而非只比较 trial 数。建议多个 seed 做重复实验。

## 输出与恢复

- manifest：完整输入指纹、分组清单、配置、压缩库指纹、关键源码指纹、依赖
  版本、GPU、模型本地路径及文件大小/mtime（不是权重内容的强哈希）。
- study：SQLite Optuna 历史，支持同配置原命令恢复；重复建议剪枝并补充未试配置。
- evaluations：每个 split 的 Full、按实际渲染文本缓存的 reuse 逐方向结果、
  汇总和耗时。长度按 bridge 加分隔符独立 tokenize，边界分词可能有差异。
- selection：测试前冻结的最终配置、验证分数与长度选择依据。
- summary：最终配置、各 split 对照、测试提升和逐 prefix 的原文/重述/拼接结果。

单目录有互斥锁；修改数据、源码、模型、预算或其他运行配置应使用新目录。
恢复失败/中断运行不会重新计算已落盘的评估。独立进程恢复的候选采样保持确定性；
GPU greedy 结果仍可能受数值精度、batch layout 和软硬件环境影响。

缓存是**分数/结果缓存**，不是跨候选常驻 donor KV 缓存。每个新文本配置仍会执行
donor/target-prefix prefill；不要据此宣称已获得推理加速或泛化提升。

## 离线验证

[tests/test_prompt_search.py](tests/test_prompt_search.py) 使用真实 Optuna 和模拟
推理后端，覆盖文本保留、目标方向、不修改输入、分组隔离、重复配置、SQLite
恢复、评估缓存、验证集长度选择与测试前冻结。它不替代真实 GPU 数值一致性验证。