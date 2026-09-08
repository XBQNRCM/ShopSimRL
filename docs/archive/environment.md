# AutoDL 训练环境备忘

2026-09-05 先在 1×A800 上装好并做过 `TP=1` smoke。当前按 **4×A800** 训练（机器若有更多卡，一键脚本只用前 4 张）。依赖在系统盘，方便保存系统镜像后换机。

算法与数据细节见 [training.md](../training.md)，W&B 口径见 [training_monitoring.md](../training_monitoring.md)。

## 1. 磁盘分工

| 位置 | 内容 | 是否进系统镜像 |
|---|---|---|
| `/root/miniconda3/envs/slime` | 训练环境（torch / TE / apex / wandb / Ray） | 是 |
| `/root/miniconda3/envs/shopsim` | ShopSimulator 服务环境 | 是 |
| `/root/sglang` | sglang 源码（editable） | 是 |
| `/root/Megatron-LM` | Megatron 源码（editable） | 是 |
| `/root/autodl-tmp/ShopSimRL` | 本仓库 | 否，换机要拷 |
| `/root/autodl-tmp/models/Qwen3.5-4B` | HF 权重 | 否，换机要拷或重下 |
| `/root/autodl-tmp/models/Qwen3.5-4B_torch_dist` | Megatron `torch_dist` | 否，换机要拷或重转 |

数据盘上 `/root/autodl-tmp/envs/slime`、`sglang`、`Megatron-LM` 只是指向系统盘的软链，给旧路径兜底。新机器不要依赖这些软链。

系统盘约剩 8G，不要再往上面堆权重或 checkpoint。数据盘约剩 33G，`TRAIN_SAVE` 必须写数据盘。

## 2. 启用环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate slime
```

激活后会通过 `etc/conda/activate.d/cudnn_ldpath.sh` 把 pip 自带的 cuDNN 9.17 放到 `LD_LIBRARY_PATH` 前面，避免落到系统 cuDNN 9.8。

关键版本（smoke 时）：Python 3.12、torch 2.11.0+cu129、CUDA toolkit 12.8、sglang 0.5.15.post1、megatron-core 0.16.0rc0、TransformerEngine 2.16.1。编译目标是 A800 / SM80，换 GPU 架构需要重编 TE / apex。

## 3. 开环境服务

训练前 ShopSimulator 要在 `http://127.0.0.1:5700` 就绪。服务用 `shopsim` 环境，不要用 `slime`。

4 卡时把槽位提到 48（一键脚本默认），`shopsim_episode_concurrency` 为 32，多出来的槽给 full-skill retry / 抖动留余量。

## 4. 训练

当前按 **4×A800**、`TP=2` 两路数据并行（切开词表 logits）。一键启动：

```bash
bash /root/autodl-tmp/ShopSimRL/scripts/run_full_train.sh
```

某一轮 slime 已写出 checkpoint、但 val/gate 还没跑时，不要清 ckpt 重来。例如 round-001 停在 iter 40：

```bash
RESUME_ROUND=1 bash /root/autodl-tmp/ShopSimRL/scripts/run_full_train.sh
```

脚本会读项目根目录 `.env` 里的 `WANDB_API_KEY`（不写进 slime 命令行），限制 `CUDA_VISIBLE_DEVICES=0,1,2,3`，拉起 ShopSimulator（48 槽），再一口气跑完 5 个外循环：每个循环 20 个 slime step，然后用 4 卡 TP=1/DP=4 服务跑 bare val + random-masked gate，注入下一轮 SkillBank。总共 100 step。每步 over-sample 32 个 group、留下 16 个（128 条 episode），`GLOBAL_BATCH_SIZE=64` 做两次更新。训练期间 sidecar 用百炼 Failure Analyst 消化 full-skill retry；round 结束再 compile。Megatron checkpoint 每轮只留最新一份，并覆盖导出同一目录的 HF 权重供 val 服务。评测并发 32、`WANDB_GROUP=qwen35-4b-train-100`。

`NUM_GPUS` 必须被 `TP_SIZE` 整除。当前默认 `TP_SIZE=2`、训练打包 `MAX_TOKENS_PER_GPU=16384`、单条封顶 `ROLLOUT_MAX_CONTEXT_LEN=24576`、单轮生成 `ROLLOUT_MAX_RESPONSE_LEN=4096`（训练与本轮 val/gate 共用，见 [runtime_eval.md](../runtime_eval.md)）。词表 logits OOM 的推演与旋钮含义见 [oom_qwen35_4b.md](oom_qwen35_4b.md)。

## 5. W&B

key 放在 `.env`，不要写进仓库或脚本参数。`rollout/shopsim/*` 随 slime 每个 step 上报。Analyst / gate 另加 `--wandb`，并用同一个 `WANDB_GROUP`。

## 6. 换机

1. 恢复本系统镜像（含 slime / shopsim / sglang / Megatron）。
2. 把 `ShopSimRL` 放到 `/root/autodl-tmp/ShopSimRL`。
3. 把 `models` 放到 `/root/autodl-tmp/models`（或重下 HF 再转 `torch_dist`）。
4. `conda activate slime`，启动 ShopSimulator，按第 4 节开训。
5. 需要监控时先 `wandb login`。

`data/` 和 `runs/` 不在 git 里，换机要一起拷或按 [training.md](../training.md) 重建 curriculum / task jsonl。
