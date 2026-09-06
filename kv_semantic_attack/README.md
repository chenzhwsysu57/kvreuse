# KV semantic attack

本目录实现 `docs/*.txt` 中描述的攻防闭环：Attacker 生成一组
`reasoning`（仅用于记录出题前分析）和
`prefix_a/prefix_b/shared_block/question/gold_a/gold_b`；Executor 执行两个
full 和两个 cross-prefix reuse；Attack Judger 根据 reasoning 判断攻击是否
成功；Defender 提出 `reasoning/remedy`；Defend Judger 比较 remedy 前后两个
reuse 方向，只有两个方向都不再混淆才算防御成功。

每次提交最多三次。每次提交的完整 API 输入输出、API 配置、Executor 输入
输出和本地模型配置都按统一的全局 step 保存到 log 目录，例如
0001_attacker.txt、0002_executor_full_a.txt、0003_attack_judger.txt。
完整 case 和执行记录仍保存在 round JSON，用于 replay，不直接拼入角色 prompt。

开发流程：Attacker 输出先做字段和答案格式校验；通过后由 Executor 执行四种
组合，再交给 Attack Judger。Judger 的 summary 追加到攻击历史；失败时带着
最新摘要重新生成，最多三次。攻击成功后进入 Defender 阶段。

Attacker 只能增加 `reasoning` 作为分析元数据，不能增加策略类型等字段；该字段
不会进入 Executor/Judger 的任务输入。gold_a/gold_b 只填写普通选项字母（A-H）。\\boxed{...}
属于 Executor 的模型输出格式约束，由 Executor 组装进执行 prompt。
Defender 的 remedy 不设长度上限；每次测试摘要都会反馈其长度，并要求后续
在保持效果的前提下继续尝试更短的 remedy。

缓存语义探针：运行 `python scripts/probe_cached_context.py --model-size 4b`。
该实验对“完整复述前文”和“指出 cached context”两个问题分别执行
`full_a/full_b/reuse_b_to_a/reuse_a_to_b`，结果保存在 `cached_context_probe/results.json`。

运行真实模型：

```bash
/home/czw/miniconda3/envs/kvreuse/bin/python \
  -m kv_semantic_attack.run_selfplay \
  --model-size 0.6b --reasoning --debug
```
