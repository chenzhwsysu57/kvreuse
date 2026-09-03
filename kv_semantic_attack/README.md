# KV Semantic Self-Play

这是一个用于“dataset-free adversarial semantic correction discovery”的最小框架。

## 核心原则

- **Attacker 只提出 synthetic task conflicts**
- **真实 Qwen3 target system 决定 attack 是否成立**
- **Defender 只提出 correction candidates**
- **真实 target system 决定哪个 correction 更好**
- benchmark 数据集不进入 discovery loop，保留到最终 frozen evaluation

## Attack validity gate

一个 attack 至少满足：

1. source task full-prefill 成功（可选但推荐）
2. target task full-prefill 成功
3. 同一个 shared block 在 KV reuse 条件下失败

即：

```text
clean(source) = success
clean(target) = success
reuse(source -> target) = failure
```

这避免把 target model 本来不会做的题误判为 KV reuse failure。

## 接入你现有仓库

只需要实现：

```python
class MyQwenExecutor(KVReuseExecutor):
    def run_full_prefill(...):
        ...

    def run_kv_reuse(...):
        ...
```

`run_kv_reuse` 内部负责你已有的：

- Qwen3 ModelScope 1.7B
- source prefix prefill
- shared block KV reuse
- RoPE correction
- post-fix correction insertion
- target decoding
- scoring

攻防系统无需知道这些内部细节。

## 推荐第一版实验设置

- attacker proposals / round: 8
- accepted verified attacks / round: 4
- defender candidates / round: 8
- rounds: 4–8
- replay buffer: 32–64 cases
- initial correction: empty string

## 推荐必须保留的对照

1. Zero-shot defender：不给 execution failures，直接生成 correction
2. One-shot failure-aware defender：只看一批 failure，不迭代
3. Random attack：不用 adaptive attacker
4. Full self-play：adaptive attack + replay + defender refinement

最终 correction 冻结后，再在 10 个真实数据集上评测。
