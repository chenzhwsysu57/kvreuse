# 可程序验证的 A/B 共享资料任务

入口：[generate_synthetic_tasks.py](generate_synthetic_tasks.py)。
实现：[synthetic_tasks.py](synthetic_tasks.py)。仅依赖 Python 标准库，不调用 LLM、GPU 或 tokenizer。

## 使用

在仓库根目录执行（Python 3.10+）：

```sh
# 默认 11 类各 100 对，共 1100 对；不是 1100 条已验证的攻击。
python kv_semantic_attack/generate_synthetic_tasks.py --per-type 100 --seed 20260906

# 指定子集，并单独覆盖某类数量。
python kv_semantic_attack/generate_synthetic_tasks.py \
  --types field_switch condition_switch label_mapping \
  --per-type 200 --count label_mapping=500 \
   --output-dir kv_semantic_attack/generated_tasks_custom

# 查看类型；控制资料规模与布局。
python kv_semantic_attack/generate_synthetic_tasks.py --list-types
python kv_semantic_attack/generate_synthetic_tasks.py --per-type 100 \
  --min-rows 8 --max-rows 16 --layouts table json \
   --output-dir kv_semantic_attack/generated_tasks_longer
```

输出目录包含 `all.jsonl`、每类一个 JSONL、`manifest.json`。默认写入
`kv_semantic_attack/generated_tasks/`，由此目录内的 `.gitignore` 忽略完整生成数据。
默认拒绝覆盖非空目录；`--overwrite` 只允许覆盖生成器自己的输出文件集合。
`--per-type` 为每类 **A/B 对数**，一对对应 A→B、B→A 两个复用方向。
允许某类 `--count TYPE=0`；总量必须大于零。

## 类型

| 类型 | 唯一变化 | 资料与答案 |
|---|---|---|
| field_switch | 最小 cost / 最小 speed | 属性表 → 行 ID |
| condition_switch | red / blue 对象中选最大 value | 属性表 → 行 ID |
| scope_switch | 前半 / 后半中选最大 value | 偶数行表 → 行 ID |
| extremum_reverse | 最小 / 最大 value | 数值表 → 行 ID |
| aggregation_switch | value 求和 / 数据行计数 | 数值表 → 整数 |
| sorting_reverse | value 升序 / 降序 | 数值表 → 逗号分隔 ID |
| priority_switch | cost 优先 / speed 优先，另一字段打破平局 | 有真实主键平局的表 → 行 ID |
| task_switch | 求和 / 原序提取操作数 | 加法表达式 → 整数 / 数字列表 |
| label_mapping | true→A / true→B，false 映射相反 | 同一阈值判断 → A/B |
| format_switch | JSON / CSV | 相同行的相同字段 → 不同序列化 |
| case_switch | 大写 / 小写 | 相同 name → 不同大小写 |
| set_relation | 两集合交集 / 差集 | 两个命名集合 → 项目列表 |
| boolean_logic | $p \land q$ / $p \land \neg q$ | 真值表 → 行 ID |
| lookup_direction | code→name / name→code | 双向字典 → 映射值 |
| counting_property | 属性为真 / 为假 的行数 | 布尔属性表 → 整数 |
| record_consistency | 左右代码差 1 位 / 差 2 位 | 成对代码 → 行 ID |

每对随机交换 A/B；同一对的措辞前导模板相同，避免引入额外方向差异。
资料行数、数值、ID、目标行和布局随机变化；提供三种前导措辞。
表达式只使用表达式布局（元数据记为 `lines`），不人为套 JSON/表格。
`scope_switch` 仅生成偶数行；指定区间中没有偶数则报错。
当前是受控基础任务，不宣称覆盖所有自然语言任务；前导措辞变化也不等于丰富的语义改写。

## 单条记录与接入

保留仓库统一字段：`task_id`, `dataset`, `prefix_a`, `prefix_b`, `shared_block`,
`question`, `gold_a`, `gold_b`, `metric`。额外包含：

- `case_id`：与 `task_id` 相同。
- `attack_type` / `dimension` / `tags`：程序配置决定的任务标签，不是对模型失败原因的归因。
- `shared_data_id`：结构化共享资料的 SHA-256，切分数据时的分组键。
- `metadata`：种子、版本、资料、两条可执行规则、布局、资料行数、重试次数。
- `answer_mode: raw`：保持原始最终回答的内容、大小写及内部空白。

`solve(data, rule)` 自动计算答案。生成阶段拒绝空筛选、单选歧义、相同答案、重复共享资料；
`validate_generated_record(record)` 再验证规则、指令、资料与答案一致性。
`generate_tasks({"field_switch": 100, "format_switch": 100}, seed=42)` 可直接在攻防框架中调用。

推理输入只能包含任务字段，不得泄漏 gold、规则或 metadata。
同一记录只有一个 `shared_block`：分别置于 `prefix_a` / `prefix_b` 后；
suffix 仍固定在移植共享块之后、`question` 之前。

### 重要：现有推理入口还不能直接正确评估所有类型

新增 [eval_synthetic_batch.py](eval_synthetic_batch.py) 可进行 Full/直接 Reuse 批量对照：
它通过作用域内适配复用 `scripts/batch_eval.py` 的批处理函数，使用仓库既有 no-reasoning
协议（关闭 explicit reasoning、`boxed_output=True` 和未完成 assistant boxed 前缀），退出时恢复原函数；不修改现有推理源文件或 self-play。

```sh
python kv_semantic_attack/eval_synthetic_batch.py --model 1.7b
# 只跑 reuse，不重复跑 full；默认 batch=16 对（32 个方向）
python kv_semantic_attack/eval_synthetic_batch.py --method reuse
# 去掉 shared_data 边界标签；原始数据不变，评测时仅删除两个标签字符串。
python kv_semantic_attack/eval_synthetic_batch.py --strip-shared-data-tags
# 小规模测试，每类两对
python kv_semantic_attack/eval_synthetic_batch.py --per-type-limit 2 \
   --output kv_semantic_attack/eval_outputs/smoke.json
```

默认按快速迭代运行：batch=16、`--verify-per-type 0`、`--save-every 0`，
不进行额外单条生成，每批只打印该批得分，结束后才汇总并保存完整结果。
`--method full/reuse/both` 控制实际执行方法；单方法不会暗中运行另一方法，
报告中的未测指标与 Full−Reuse 对照记为 null。默认输出文件名包括方法、模型及 batch。
可用 `--save-every N` 每 N 批保存一次累计进度；默认中断前不会定期保存。
需严格检查时，显式使用 `--method both --verify-per-type 1`：
每类首对检查真正单方向 batch-size-one 与批量的生成 token 一致性，
并检查同前缀缓存复用与 full 一致性；不一致时保存未完成报告并停止。
`--allow-batch-mismatch` 可记录批量偏差后继续，但同前缀检查仍必须通过。
关闭检查仅减少重复计算，不代表之前发现的批量/单条差异已解决。
`--strip-shared-data-tags` 在验证原数据后删除边界标签，保留换行、资料、指令和答案，
报告保存此开关及实际输入指纹；默认报告名增加 `no_tags`，不覆盖有标签结果。
比较标签影响时应使用相同 batch 大小，避免把批量数值差异误判为标签效应。
不添加纠偏 suffix，不重算共享块；保留直接复用所需的 RoPE 位置迁移。
耗时包括 donor KV 构建，不代表线上缓存命中加速比。

现有 `scripts/batch_eval.py` 及攻防执行器的 boxed 输出设定、选项答案提取不适用于这些任务，
尤其格式切换、大小写、完整排序列表。后续接入需：

1. 保持仓库的 no-reasoning boxed assistant 前缀；评测器会提取平衡的 boxed 内容。
2. 使用 `score_response(record, side, response)` 对提取的 boxed 内容严格判分：只去除输出首尾传输空白，
   不小写化、不抽取 boxed、不丢弃前后解释。JSON 空格等内部格式也严格匹配。
3. tokenizer 渲染后检查 A/B 共享块 token ID 一致；本生成器仅保证共享文本一致，
   不声称检查过模型分词或位置处理。
4. 每类先测 Full，分别报告整体表现和 Full 正确子集中的 reuse 失败率。
   自动生成的任务对不一定会导致 KV reuse 失败。

## 可复现性与数据隔离

固定代码版本、参数、seed 得到相同数据和稳定 ID；每类独立随机流，改变其他类数量不改变该类数据。
`manifest.json` 保存生成参数和计数。不自动划分搜索/验证/测试；
按 `shared_data_id` 分组，并将该资料的所有改写和双向推理放在同一集合。
未来若增加资料重新排序等变体，应显式保留其父资料分组 ID。
若需更强泛化评估，应再保留未参与搜索的任务规则、模板或组合，不能仅换 seed 就声称跨任务泛化。

## 自适应攻击：第一步

[adaptive_attack.py](adaptive_attack.py) 实现“攻击 LLM → 程序生成”的受限接口。
攻击 LLM 只可提出 `task_weights`、样本数、资料长度和布局；程序负责验证配置、分配每类数量、
生成资料和 gold。LLM 不能提交自由题目、答案、suffix 或代码。

```sh
cat > /tmp/attack_distribution.json <<'EOF'
{"task_weights":{"task_switch":3,"boolean_logic":1,"set_relation":1},"pairs":128,"min_rows":4,"max_rows":12,"layouts":["table","json"],"min_pairs_per_type":4,"rationale":"focus on observed weak families"}
EOF
python kv_semantic_attack/generate_adaptive_attack_batch.py \
   --proposal /tmp/attack_distribution.json --seed 20260907 --round 1 \
   --output kv_semantic_attack/adaptive_runs/round_01_candidates.jsonl
```

数量采用最大余数法，`min_pairs_per_type` 是探索下限；生成数据附加不可变的 distribution、seed、round 和 request ID。
该步骤不调用模型、也不筛选有效攻击。下一步才评估 `Full correct ∧ Reuse+suffix wrong`。

调用真实 Qwen 3.8 Max 时使用 [run_adaptive_attacker.py](run_adaptive_attacker.py)。
它复用 `.env.local` 的 `DASHSCOPE_*` 配置，并将完整请求／响应写入 `--log-dir`：

```sh
python kv_semantic_attack/run_adaptive_attacker.py \
   --dashboard /tmp/dashboard.json --seed 20260907 --round 1 \
   --pairs 1000 \
   --output kv_semantic_attack/adaptive_runs/round_01_candidates.jsonl --debug
```

攻击 LLM 的分析必须置于 JSON 的 `reasoning` 字段；程序只接受指定 schema，随后独立生成资料及 gold。
`--pairs N` 可在 LLM 已选择任务权重、行数和布局后强制本轮总量为 $N$ 对（1–2000）；
例如 `--pairs 1000` 生成 1000 对、2000 个复用方向。

## 环境 step 写盘

三个阶段都支持 `--run-dir RUN_DIR --step N`（必须同时提供）。每次动作创建不可覆盖的
`RUN_DIR/steps/step_NN_*.json`，记录输入／输出路径与 SHA-256、模型与 batch 配置、攻击分布、
每个 suffix 全文及其分类型结果和耗时。API 的完整请求与原始回答写入 `RUN_DIR/api_traces/`。

```sh
# step 1：攻击分布和生成数据；step 2：防御suffix；step 3：Full/Reuse/suffix评测
python kv_semantic_attack/run_adaptive_attacker.py ... --run-dir kv_semantic_attack/adaptive_runs/run_001 --step 1
python kv_semantic_attack/run_adaptive_defender.py ... --run-dir kv_semantic_attack/adaptive_runs/run_001 --step 2
python kv_semantic_attack/evaluate_suffix_candidates.py ... --run-dir kv_semantic_attack/adaptive_runs/run_001 --step 3
```

同一 step/action 已存在时程序拒绝覆盖；每轮继续使用下一个 step 编号。

## 多 GPU 防御评测

[run_multi_gpu_defense.py](run_multi_gpu_defense.py) 将 Full、direct Reuse 和每个 suffix 作为独立进程调度。
`--gpu-slots` 指定每张 GPU 最多并发任务数；每个子进程退出后会释放模型和 CUDA 缓存。

```sh
python kv_semantic_attack/run_multi_gpu_defense.py \
   --input kv_semantic_attack/adaptive_runs/run_001/round_01_candidates.jsonl \
   --candidates kv_semantic_attack/adaptive_runs/run_001/round_01_suffixes.json \
   --output-dir kv_semantic_attack/adaptive_runs/run_001/parallel_eval \
   --gpu-slots 0=2 1=2 2=1 \
   --max-used-gib 4 --model 1.7b --batch-size 16
```

启动任务前，空闲 GPU 的外部显存占用必须不超过 `--max-used-gib`；调度器无法从 `nvidia-smi` 区分自己的子进程，
因此一旦已占有一个 slot，就按用户授权填满该 GPU 的剩余 slot。每个 job 的终端输出保存在 `output-dir/logs/`，
结果和 GPU、退出状态、耗时保存在 `dispatch_manifest.json`。

## 固定攻击池的五轮防御优化

入口为 [run_defense_refinement.py](run_defense_refinement.py)。本次改动只完善流程，
没有自动运行新 API 请求或模型实验。

### 防御者的任务边界

实际顺序为 **target prefix → source-conditioned shared-block KV → suffix → question**。
目标任务规则位于 target prefix；question 可能只要求返回选项字母。因此 suffix 必须保留
当前目标规则，不能宣称所有前置指令失效、只有后续 question 才定义任务。
suffix 只是新增文本，不能声称已经清空、修改或重新计算共享 KV。
诊断示例中的 gold 只供防御者分析，不得写入通用 suffix 或评测任务输入。

每个候选仍采用 `candidate_id/reasoning/suffix` schema，英文 suffix 最多800字符。
reasoning 应包括：观察证据、尚未证实的原因假设、相对历史方案的变化、预期修复和退化风险。
不要求本地 no-reasoning 模型输出分析；防御 API 每个候选分配1200个输出 token 的总预算，
例如4个候选使用 `max_tokens=4800`。客户端仍不开启 thinking。

### 每轮反馈和选择

- 默认 `--max-attempts 5`、每轮4个新候选；固定输入池，不同时更新攻击分布。
- 失败反馈每轮按当前所选 suffix 的 **Full accuracy − Reuse accuracy** 降序排列类别，
   差距相同时优先样本数较多者，再按类别名确定性排序。不用样本量重新加权差距。
   `failure_priority` 显式列出类别、差距（百分点）、方向数和有效失败数，防御者须考虑小样本不确定性。
- 当前所选候选每类默认最多6个 Full正确、Reuse错误案例；两个方向均足够时各3个，
   不足时由其他方向补足。类内兼顾资料布局、共享文本长度（500字符分桶）和源答案泄漏/其他错误，
   按 task_id/target 去重，采样不依赖输入排列。保留全部类别统计，不丢弃低差距类别。
- 历史候选也按各自差距排序抽取修复/退化/持续失败案例，分别保留各类变化的预算；
   退化案例不会因其类别差距低而全部被持续失败案例挤掉。持续失败示例仅取Full正确者，
   变化总数仍统计所有方向。单个有效失败都是正确性从1降到0，不再虚构样本级准确率差距排序。
- 每轮重新计算 Full、无防御 Reuse、新候选及当前历史最佳，保证相同评测配置下比较。
- 下一轮收到 `refinement_history`：所有历史候选全文、设计理由、整体/分类型指标、
   相对无防御和 incumbent 的修复/退化数，以及每候选每比较对象每种变化最多2个案例。
   所有失败候选也保留，不只反馈胜出者；案例标明 Full 是否正确，避免混淆任务难度与复用失败。
- 小于3个百分点的改善也保留为下一轮 incumbent；阈值不阻止搜索更新。
- `--minimum-improvement-pp 3` 表示最终目标：相对第1轮在本配置实测的初始 incumbent，
   累计提升至少3个百分点。默认即使达标仍跑满5轮；仅显式 `--stop-on-threshold` 才提前退出。
- 每轮新候选允许1–7个，为重测 incumbent 预留第8个位置。
- run-dir 必须新建或为空；原始 API 候选文件不被补入 incumbent 的操作改写，另存
   `refine_NN_evaluated_suffixes.json`，保持 step manifest 引用的哈希有效。
- 最后写 `refinement_summary.json` 及 `accepted_suffix.json` 或 `refinement_stopped.json`。
   每轮 dashboard 含完整搜索历史，可供检查和后续设计；不自动续跑已有目录。

五轮入口可传 `--gpu-slots 1=2 2=2 3=2`，每轮调用多 GPU 调度器；省略时保持单进程评测。
每轮4个新候选加Full和无防御Reuse为6个任务；需要额外重测incumbent时第7个任务排队，
始终不超过每卡2任务。轮次仍串行推进，上一轮完整反馈产生后才生成下一轮候选。
`run_multi_gpu_defense.py --output ...` 会验证6/7份结果的配置、输入指纹、候选文本和方向覆盖，
生成五轮反馈所需的统一evaluation；任何任务失败或合并验证失败不会进入下一轮。

本机一键入口：[../scripts/run_defense5_gpu123.sh](../scripts/run_defense5_gpu123.sh)。固定GPU1/2/3各2槽、
Qwen3-1.7B、no-reasoning、batch=8、128新token，使用已有1000对池。先重测旧4个候选，
通过 `build_adaptive_dashboard.py --include-candidate-feedback` 写入第0轮历史，再跑满5轮新候选。
可用 `RUN_DIR` 指定新运行目录（拒绝覆盖），`PYTHON` 指定解释器，`INPUT/CANDIDATES` 指定输入；
每张卡空闲时默认要求外部显存占用不超过4GiB，否则等待。输出目录内pipeline.log记录全流程，
baseline_parallel和refinement/refine_NN_parallel保留逐任务日志。

### 判分口径与数据隔离

修复 `raw_result` 在第一个 token 后提前返回的问题：现在保留直到首个 EOS 的完整序列，
并对完整解码文本提取答案，空输出也返回有效结果字典。旧产物不改写；开始新五轮前应使用
修复后的判分重新评估基线，最好重新测旧候选以创建同口径初始历史。不要把旧首token得分
和新完整答案得分直接混算。旧 dashboard 可作为注明来源的背景，但不是新的接受阈值基准。

固定池结果只表示搜索集提升。五轮后应冻结最佳 suffix，再对未参与反馈的留出资料评测；
留出集按 shared_data_id 分组，不将双向记录分到不同集合。本入口不自动构造留出集。

## 离线回归测试

```sh
python -m unittest discover -s kv_semantic_attack/tests -p 'test_synthetic_tasks.py' -v
```

覆盖每类 100 对、固定种子、多种子、三种布局反向解析、已知答案 oracle、真实优先级平局、
标签阈值边界、严格输出判分、输入校验、CLI 数量覆盖及防覆盖保护。