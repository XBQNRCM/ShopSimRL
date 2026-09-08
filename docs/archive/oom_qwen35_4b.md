# Qwen3.5-4B / 4×A800 训练 OOM 备忘

2026-09-05 在 `run_full_train.sh` 200-step 作业上连续三次 `torch.OutOfMemoryError`，都发生在 Megatron 训练 forward 的 **vocab softmax / log-prob**，不是 rollout、不是 ShopSimulator。本文记录现象、原因和最终旋钮，避免把几个看起来都像「显存」的开关当成一回事。

硬件：4×A800-SXM 80GB。模型：Qwen3.5-4B，`text_config.vocab_size=248320`（约 3/4 层 linear attention）。训练 colocated：Megatron 与 SGLang 分时占用同一组 GPU。

## 1. 三次崩溃分别卡在哪

| 次序 | 当时配置 | 报错点 | 额外申请 | 当时占用 |
|---|---|---|---|---|
| 1 | `TP=1`，`MAX_TOKENS_PER_GPU=32768` | logits `float16_to_fp32` | ~29.5 GiB | 单卡空闲不够一整份 `[T,V]` fp32 |
| 2 | `TP=1`，打包降到 8192，仍无 chunk | softmax 复制整份 `[T,V]` | ~13 GiB | 单条超长 episode 独占 micro-batch，打包上限管不到 |
| 3 | `TP=2`，`--log-probs-chunk-size 1024` | 仍在 softmax | 464 MiB | 77.3 / 79.3 GiB，空闲 207 MiB |

第三次说明：**分块只压瞬时峰值，不压 backward 保留量。** slime `ppo_utils._VocabParallelLogProbEntropy` 每个 chunk 算完 softmax 都 `save_for_backward(log_prob_softmax, …)`。8 个 chunk 各存 1/8，加起来仍是完整的 `[T, V/TP]`。TP=2 把这份从约 8 GiB 砍到 4 GiB，chunk 把单次分配从 4 GiB 砍到 ~0.5 GiB，但 4 GiB 一直挂到反传结束。

对症开关是 `--recompute-loss-function`（`slime/backends/megatron_utils/loss.py`）：对整个 loss 段做梯度检查点，softmax 不进 `saved_tensors`，backward 时重算。梯度精确，代价是 loss 段多跑一遍 forward。不要和「截断序列」混为一谈。

不要开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 来赌碎片。sglang `torch_memory_saver` 同样走 CUDA VMM，两者有已知冲突；第三次 reserved-未分配只有 ~600 MiB，瓶颈不是碎片。

## 2. 三个长度旋钮不是一件事

一条购物 episode 在训练里会变成多条 slime `Sample`，再被打进若干 micro-batch。三个旋钮作用在不同层：

| 旋钮 | 作用对象 | 会不会截断单条 Sample | 决定什么显存 |
|---|---|---|---|
| `ROLLOUT_MAX_CONTEXT_LEN` | 生成侧 context + finalize `tokens[:cap]` | **会**。触顶且解析不出 tool call → `generation_length`，环境零分，`Sample.status=TRUNCATED`，**仍进 GRPO** | 单条 Sample 的上限，也是峰值上限 |
| `MAX_TOKENS_PER_GPU` | first-fit 把多条 Sample 装进同一 micro-batch | **不会**。单件比箱子大就独占一箱、允许超载 | 短样本箱子的容量。峰值 = `max(箱子, 本步最长 Sample)` |
| `LOG_PROBS_CHUNK_SIZE` | softmax 沿 T 切片 | 不会 | 瞬时分配；有 recompute 时不再决定保留量 |

`dp_schedule` 的不变量：除「单条超过容量」外，每个 micro-batch `≤ max_tokens_per_gpu * cp_size`。训练用 THD packed sequence（`qkv_format=thd`），注意力按各条 `cu_seqlens` 算，不是把拼箱当成一条超长序列做平方注意力。

Qwen3.5-4B、TP=2 时一份 fp32 `[T, V/2]` 大约：

| T | 一份 `[T, 124160]` fp32 |
|---|---|
| 4096 | 2.0 GiB |
| 8192 | 4.0 GiB |
| 16384 | 7.8 GiB |
| 24576 | 11.6 GiB |
| 32768 | 15.5 GiB |

无 recompute 时 backward 还要再挂一份同尺寸的 softmax。有 recompute 后这份不再常驻。

## 3. 截断、fork、打包：报表会骗人

ShopSim 适配器 `fork_threshold_tokens=0`：工具 schema / chat template / reasoning 回显造成 token drift 时直接切段，保留每一轮真实采样的 logprob。每条 fork 出来的 Sample 带的是 **从该段起点起的完整上下文**，不是增量。

`n_samples_per_prompt=8` 是同一题 8 条独立 episode（GRPO 组）。fork 是 **一条 episode 内部**再切成多条训练 Sample。两件事不要混。

`rollout/truncated_ratio` 是 `mean(status == TRUNCATED)`，分母是 Sample 数。episode 因 `generation_length` 结束时，runtime 给 **这条轨迹的每一段**都打上 `TRUNCATED`，包括前面没触顶的短段。触顶 episode 往往最长、fork 最多，所以 5% 的 episode 触顶可以显示成 10–25% 的 truncated_ratio。它仍是可评分的零分策略失败，不是被 filter 丢掉；只有 `ABORTED` + `remove_sample` 才整组丢弃。

round-000 的 256 条 episode（Qwen tokenizer，当时 context 还是 32K）：

| 区间 | n | reward>0 |
|---|---|---|
| 0–8,192 | 191 | 68% |
| 8,192–12,288 | 32 | **84%** |
| 12,288–16,384 | 12 | 17% |
| 16,384–24,576 | 8 | 62% |
| 24,576+ | 13 | 8% |

约 1100–1200 token/环境步；4–6 步中位 6.5k（80% 样本）。把 context 砍到 8192 会切在成功率最高的带上。24576 落在垃圾带边缘：牺牲约 0.6% 成功 episode，触顶率约 5%。`generation_length` 与单 turn `rollout-max-response-len=2048` 在 ShopSim 里不区分，统一零分进训练。

wandb run `jin9wfo9` 前 8 步 `total_lengths` 均值 **6.7k–8.4k**，已经大于当时的 4096 箱子。first-fit 对「单件 > 容量」是独占一箱，所以 4096 对大多数 Sample 是空转：峰值仍是 24k 独苗，短样本也没捆上。真正的分配混合是 **7k 独苗 + 24k 独苗**，不是「4k 满箱」。

## 4. 训练 TP 和推理 TP

训练 OOM 的主因是 lm_head 词表维。4 卡上 `TP_SIZE=2`（DP=2）把每卡词表切半，并打开 sequence parallel，够用。`TP_SIZE=4`（DP=1）再砍一半 logits，但每卡总 FLOPs 不变、丢掉 DP、GEMM 更瘦、all-reduce 更宽，4B 模型通常更慢。只在打算把 context 加回 32K 时才值得考虑。

Rollout / val 没有这份反传 logits。保持 `ROLLOUT_TP_SIZE=1`，4 个独立引擎。评测同样 4 卡 `tp=1 dp=4` 起 sglang，不要为训练 OOM 给推理加 TP。

## 5. 最终配置（2026-09-06）

`scripts/run_full_train.sh` / `scripts/run_shopsimrl_slime.sh` 默认：

```text
NUM_GPUS=4
TP_SIZE=2                          # 仅 Megatron
ROLLOUT_TP_SIZE=1                  # rollout / val
MAX_TOKENS_PER_GPU=16384           # 打包；峰值仍由 24k 独苗决定
ROLLOUT_MAX_CONTEXT_LEN=24576      # 单条 Sample 硬顶
LOG_PROBS_CHUNK_SIZE=1024
--recompute-loss-function
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 1
```

`MAX_TOKENS_PER_GPU=16384` 的理由：峰值公式是 `max(箱子, 最长 Sample)`。箱子只要 **不超过 24576**，就不会创造比已跑通的 24k 独苗更大的 micro-batch。16384 让两条典型 ~7k Sample 进同一箱，箱数大约减半，THD 下注意力仍按短序列算。提到 24576 也能保持峰值不变，但会把「截断率低的好步骤」的峰值从 ~12–16k 抬到 24k，边际吞吐收益更小。

**不要**把 `MAX_TOKENS_PER_GPU` 加到 24576 以上：first-fit 可能把 20k+8k 捆成 28k，峰值高于独苗。

`--balance-by-flops` 按 FLOPs 切 micro-batch，**不保证** token 上限，箱子已经很满时不要开。

改这些环境变量只在下一轮 `run_shopsimrl_slime.sh` 启动时生效（外循环 round 边界）。正在跑的 slime job 不会热更新。

## 6. 若再次 OOM

先看堆栈，不要先改 TP。

1. 仍在 `softmax` / log-prob，额外只要几百 MB：多半是碎片或 24k logits 余量见底。先确认 `--recompute-loss-function` 还在；再把 `ROLLOUT_MAX_CONTEXT_LEN` 降到 16384（会真的截断长 episode，零分进训练）。不要优先降 `MAX_TOKENS_PER_GPU`——它已经不决定峰值。
2. 一次申请十几 GiB：封顶没生效，或某条 Sample 绕过了 `max_sample_tokens`。查 `rollout/total_lengths` 的最大值，不是均值。
3. 推理阶段 / SGLang scheduler：与本文路径无关，不要用训练 TP 去「修」它。

现场可看：`rollout/truncated_ratio`（Sample 口径，有 fork 放大）、`perf/actor_train_time`（24k 独苗多的步骤会到 500s+）、`perf/train_wait_time`（colocated 下这是 rollout，GPU 通常仍在忙）。`wait_time_ratio≈30%` 不是 GPU 空闲率。
