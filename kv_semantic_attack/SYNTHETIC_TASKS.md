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

## 测试

```sh
python -m unittest discover -s kv_semantic_attack/tests -p 'test_synthetic_tasks.py' -v
```

覆盖每类 100 对、固定种子、多种子、三种布局反向解析、已知答案 oracle、真实优先级平局、
标签阈值边界、严格输出判分、输入校验、CLI 数量覆盖及防覆盖保护。