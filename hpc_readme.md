# kvreuse HPC 使用说明

本文说明如何在 GPU 计算节点上从 Git 仓库开始，下载数据集、构造验证集，并运行 Qwen3 的 full-recompute 与 direct KV-reuse 实验。

## 1. 克隆与环境

将 `<REPO_URL>` 替换为实际仓库地址：

```bash
git clone <REPO_URL> kvreuse
cd kvreuse
git checkout main
conda create -n kvreuse python=3.10 -y   # 已有环境时跳过
conda activate kvreuse
```

安装依赖。PyTorch 的 wheel 必须与集群 CUDA 版本匹配：

```bash
python -m pip install -U pip
python -m pip install numpy matplotlib transformers modelscope
# 仅在环境中没有兼容 torch 时安装，例如 CUDA 12.1：
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
nvidia-smi
```

登录节点适合下载和构造数据；模型推理必须提交到 GPU 节点。

## 2. 镜像和模型缓存

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

数据集下载器默认使用 Hugging Face 镜像。模型由 ModelScope `snapshot_download` 获取。若模型已经在 `legacy/modelscope_models`，可先复制/移动到 ModelScope 缓存目录；否则首次运行时加 `--allow-download` 自动下载。正式离线运行时省略该选项。确认 `model.safetensors.index.json` 所列分片全部存在。

推荐在联网节点预下载模型：

```bash
conda run --no-capture-output -n kvreuse \
  python -u scripts/download_models.py --models 0.6b 1.7b 4b 8b
```

也可以只下载部分模型：

```bash
conda run --no-capture-output -n kvreuse \
  python -u scripts/download_models.py --models 1.7b 4b
```

脚本会在每个模型下载完成后检查权重分片和 `.incomplete` 文件。预下载完成后，推理命令不需要 `--allow-download`，不会在 GPU 作业中等待网络。

## 3. 下载原始数据集

```bash
conda run -n kvreuse python scripts/download_datasets.py
```

只下载新增数据集：

```bash
conda run -n kvreuse python scripts/download_datasets.py \
  --datasets pku_safe_rlhf harmbench_contextual
```

原始文件在 `data/raw/`，版本、URL、大小和 SHA-256 在 `data/raw/manifest.json`。已有非空文件会复用；强制重下才使用 `--force`。

## 4. 构造和验证

```bash
conda run -n kvreuse python scripts/build_datasets.py \
  --seed 20260827 --max-per-dataset 1000
conda run -n kvreuse python scripts/validate_datasets.py
```

输出为 `data/processed/all.jsonl`、按数据集拆分的 JSONL，以及 `data/processed/examples/` 样例。记录字段为 `task_id, dataset, prefix_a, prefix_b, shared_block, question, gold_a, gold_b, metric`。过滤器去掉答案不变、最优项不唯一或无法自动评分的记录；HelpSteer2 只保留 0/4 极端标签。

## 5. 共同验证集（每类 10 条）

```bash
conda run -n kvreuse python scripts/select_validation_subset.py \
  --input data/processed/all.jsonl \
  --output data/validation/second_step_10_each_harmbench.jsonl \
  --per-dataset 10 --seed 20260827
```

应得到五个数据集各 10 条，共 50 条；四个模型必须使用同一个文件。

## 6. Full-recompute 基线

```bash
conda run -n kvreuse python scripts/run_full_recompute_validation.py \
  --model 1.7b \
  --input data/validation/second_step_10_each_harmbench.jsonl \
  --output-root results/full_recompute_validation \
  --retry-truncated --retry-max-new-tokens 1024
```

`--model` 可选 `0.6b`、`1.7b`、`4b`、`8b`。汇总：

```bash
conda run -n kvreuse python scripts/summarize_full_recompute_validation.py \
  --root results/full_recompute_validation
```

## 7. Direct KV-reuse

每条记录运行 `full_A/full_B` 和跨 prefix 的 `A→B/B→A`。共享 block 的 Key 先逆源位置 RoPE，再应用目标位置 RoPE；Value 直接复用。热图使用 `viridis`（0 深紫，1 黄色）。

```bash
conda run -n kvreuse python scripts/run_direct_reuse.py \
  --model 1.7b \
  --input data/validation/second_step_10_each_harmbench.jsonl \
  --output-root results/direct_reuse --max-new-tokens 512 \
  --retry-truncated --retry-max-new-tokens 1024
```

结果位于 `results/direct_reuse/qwen3-<model>/`：`samples.jsonl` 保存输入/输出和指标，`summary.json` 保存汇总，`samples/<task_id>/kv_similarity.npz` 保存矩阵，`kv_similarity.png` 保存图像。

## 8. Self-control

代码或 Transformers/attention 后端变化后，先验证 `A→A`、`B→B`：

```bash
conda run -n kvreuse python scripts/run_direct_reuse.py \
  --model 1.7b --input data/validation/second_step_10_each_harmbench.jsonl \
  --output-root results/direct_reuse_selfcheck --max-samples 10 \
  --run-self-controls
```

Self-control 通过后，正式实验可省略它们以避免重复运行 full baseline。

## 9. 一键运行 reasoning ablation

默认使用 1.7B 和每类 10 条（共 50 条）验证集，依次运行 no-reasoning 与 explicit-reasoning：

```bash
bash scripts/run_reasoning_ablation.sh
```

也可以指定模型和输入文件：

```bash
bash scripts/run_reasoning_ablation.sh 4b data/validation/second_step_10_each_harmbench.jsonl
```

两组结果分别写入 `results/reasoning_ablation/no_reasoning/` 和 `results/reasoning_ablation/reasoning/`。

对当前 `data/processed/all.jsonl` 的全部构造数据运行四个模型：

```bash
bash scripts/run_all_models_reasoning_ablation.sh
```

也可传入另一个 JSONL 路径作为输入。该脚本会依次运行四个模型的 origin/reuse × no-reasoning/reasoning，结果仍按模型和模式分目录保存。

## 10. Slurm 示例

```bash
#!/bin/bash
#SBATCH --job-name=kvreuse-17b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kvreuse
cd /path/to/kvreuse
export HF_ENDPOINT=https://hf-mirror.com
python scripts/run_direct_reuse.py --model 1.7b \
  --input data/validation/second_step_10_each_harmbench.jsonl \
  --output-root results/direct_reuse --max-new-tokens 512 \
  --retry-truncated --retry-max-new-tokens 1024
```

提交：`sbatch run_kvreuse.slurm`。

## 11. 常见问题

- `CUDA is required`：将模型测试提交到 GPU 节点。
- `model cache is incomplete`：补齐 safetensors 分片后再运行。
- `hit max_new_tokens`：使用 `--retry-truncated`；若仍触发，应检查输出是否进入自我纠错循环。
- 无网络：在联网节点下载 `data/raw/` 和 ModelScope 缓存，再复制到共享盘；离线运行不要加 `--allow-download`。
- 可复现性：固定 seed、复用同一个 validation JSONL，并保留 `run_config.json`、`samples.jsonl`、`summary.json`。
