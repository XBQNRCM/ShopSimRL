<div align="center">
<img src="assets/logo.png" alt="ShopSimulator Logo" width="200"/>
<h1>ShopSimulator</h1>
<p>Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants</p>
</div>

<p align="center">
<a href="https://arxiv.org/abs/2601.18225">Paper</a> ·
<a href="https://huggingface.co/datasets/wpei/ShopSimulator">Dataset</a> ·
<a href="https://github.com/ShopAgent-Team/ShopSimulator">Upstream</a>
</p>

本目录提供 ShopSimRL 使用的本地购物环境。当前分支已统一数据、搜索、奖励和 observation 契约；Agent runtime、轨迹采样与 checkpoint 评测位于项目根目录的 `shopsimrl/`，不再保留旧的 `single_eval/`、`multi_eval/` 和逐 task JSON 评分入口。

## 目录

```text
ShopSimRL/
├── ShopSimulator/shop_env/  # 环境、cleaned 数据与搜索索引
├── shopsimrl/               # 可复用 Agent runtime 与评测 pipeline
├── configs/                 # 实验配置
├── scripts/                 # 采样/评测入口
└── tests/                   # runtime 与 pipeline 测试
```

## 启动环境

Linux/macOS：

```bash
cd shop_env
bash setup.sh
bash start.sh
```

Windows PowerShell：

```powershell
cd shop_env
.\setup.ps1
.\start.ps1
```

服务默认位于 `http://127.0.0.1:5700`。安装后可运行不依赖 LLM 的 smoke test 和单测：

```powershell
python scripts\smoke_test.py --task 0
python -m unittest discover -s tests -v
```

浏览器访问 `http://127.0.0.1:5700/debug-ui` 可以从 Agent 视角手动检查搜索、页面可点击元素、规格选择、终止和奖励。访问 `http://127.0.0.1:5700/replay-ui` 并输入 `runs\<experiment>`，可以查看整批实验概览并逐 task 回放 Agent 的 thinking、tool call 与环境反馈。完整环境契约见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)，回放设计和 artifact 语义见 [../docs/replay_ui.md](../docs/replay_ui.md)。

## 数据与固定划分

环境只接受 cleaned v2 商品档案和对应 manifest。准备运行时 JSON 与搜索索引：

```powershell
cd shop_env
python scripts\prepare_data.py
```

项目 persona 划分固定为 `train=3726`、`val=400`、`test=400`。重新核验或显式生成相同的 split manifest：

```powershell
python scripts\build_project_splits.py
```

`task_splits.cleaned.v2.json` 只记录清洗后样本的上游来源；训练与评测必须读取 `persona_splits.v1.json`。

## Agent runtime 与评测

从 ShopSimRL 项目根目录运行：

```powershell
python scripts\run_shopsimrl.py plan configs\qwen35_4b_val.yaml
python scripts\run_shopsimrl.py run configs\qwen35_4b_val.yaml
```

Qwen3.5-4B 的 train/val/test 配置已分别固化在 `configs/qwen35_4b_*.yaml`，统一启用 thinking mode 和标准 function tool calling；采样和评测使用同一入口。例如运行冻结测试集：

```powershell
python scripts\run_shopsimrl.py plan configs\qwen35_4b_test.yaml
python scripts\run_shopsimrl.py run configs\qwen35_4b_test.yaml
```

设计与 artifact 约定见项目根目录的 `docs/code_plan.md`。

## Citation

```bibtex
@misc{wang2026shopsimulatorevaluatingexploringrldriven,
  title={ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants},
  author={Pei Wang and Yanan Wu and Xiaoshuai Song and Weixun Wang and Gengru Chen and Zhongwen Li and Kezhong Yan and Ken Deng and Qi Liu and Shuaibing Zhao and Shaopan Xiong and Xuepeng Liu and Xuefeng Chen and Wanxi Deng and Wenbo Su and Bo Zheng},
  year={2026},
  eprint={2601.18225},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2601.18225}
}
```
