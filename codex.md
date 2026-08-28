请为我实现一个用于研究 **KV cache reuse 在 prefix / agent role 改变时是否损害模型重新理解共享文本 block 能力** 的实验框架。模型使用 ModelScope 上的 Qwen3-0.6B、Qwen3-4B、Qwen3-8B。必须使用 Qwen3 官方 chat template；system 固定为 `You're a helpful assistant.`，除此之外的 prefix、共享 block、问题和输出要求全部放在同一个 user message 中。

请优先构造并接入以下任务：① 对抗式语义任务：IBM ArgKP，构造成同一组 arguments 在 `pro` 与 `con` prefix 下选择不同论据；② 对抗式 utility 任务：Deal or No Deal，根据不同 agent 的 utility function 从同一候选 allocation block 中选择最优项；③ 多 rubric Judge：HelpSteer2，同一个 prompt-response block 分别按 correctness、helpfulness、coherence、complexity、verbosity 评分；④ 冲突 Judge：PKU-SafeRLHF，对同一 response pair 分别按 helpfulness 和 safety 选择；⑤ Safety intent：使用 HarmBench contextual 数据，把 `ContextString` 作为共享 block，构造 benign intent 与 harmful/adversarial intent 两种 prefix。所有任务尽量转成稳定、可自动评测的分类、选择、排序或离散打分问题，避免依赖额外 LLM judge。

请统一生成类似下面的数据结构：`task_id, dataset, prefix_a, prefix_b, shared_block, question, gold_a, gold_b, metric`。首先验证每个样本确实满足“相同 shared_block 在不同 prefix 下 gold 或信息重要性发生变化”；过滤掉两个 prefix 下答案相同、歧义过大或无法自动评分的样本。实现数据下载、预处理、缓存和可复现采样，并为每个数据集生成少量可人工检查的样例 JSONL。

随后实现两种推理条件并进行严格对比：`full_recompute` 表示 prefix 改变后正常重新计算整个 user sequence；`kv_reuse` 表示共享 block 的 KV 沿用其在另一 prefix 下计算得到的缓存，仅对必要的新 prefix / query 部分重新计算。请特别检查 token position、RoPE position id、attention mask 和 Qwen3 cache API，确保比较真正对应研究问题，而不是由位置错位等实现 bug 导致。记录每个样本两种条件下的预测、logits / token probabilities、正确性和性能指标，并输出按 model、dataset、task type 汇总的性能差值 `full_recompute - kv_reuse`。

请把代码组织成可直接运行的项目，至少包含数据预处理脚本、统一 dataset schema、Qwen3 inference / KV manipulation 模块、evaluation 脚本、配置文件和 README。优先先实现最小可运行版本：ArgKP、HelpSteer2 和 Deal or No Deal；确认 pipeline 正确后再扩展 PKU-SafeRLHF 和 HarmBench。不要只写方案说明，请直接实现代码，并在 README 中明确指出任何无法通过 Transformers / ModelScope 公共 API 精确实现的 KV reuse 部分以及采用的替代实现。

第一步，请先下载数据集、以及想办法构造数据集。

第二步，对于每类数据集，写代码，跑10条qwen3-0.6b, qwen3-4b, qwen3-8b的数据看看。保留输入、输出，看没有kv reuse的情况下的输出是什么样子的？这一步是为了验证数据集的有效性。在反复测试通过之后，确认数据集构造没问题，我们才会进行下一步reuse的实验。