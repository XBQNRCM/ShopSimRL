# 复现、检查与扩展

项目页是可下载数值证据的静态研究报告，主要内容无需访问源仓库即可阅读。[资料库](../library.html)在站内渲染源文档，原始中文工程文档明确标注语言，不静默机器翻译。

## 重建冻结分析

在仓库根目录使用 Python 3.10 或更新版本：

```bash
python -m pip install -e ".[analysis,site]"
python scripts/analyze_training.py
python scripts/analyze_behavior.py
python scripts/analyze_wandb.py
python scripts/plot_training.py
python scripts/plot_behavior.py
python scripts/plot_wandb.py
python scripts/build_site.py
python scripts/check_publication.py
python -m http.server 8000 --bind 127.0.0.1 --directory _site
```

打开 `http://127.0.0.1:8000`。默认聚合只读取提交的数值摘录与冻结 gate，不请求模型或 W&B。行为 bootstrap 使用 5,000 次任务重采样，主 test 区间沿用 comparison 实现的 10,000 次，两者分别记录随机数约定。

## 刷新原始测量

`analyze_training.py --refresh-local` 读取本地训练日志、triage 和评测 trace。`analyze_behavior.py --refresh-local --tokenizer /path/to/tokenizer.json` 另外解析全部 turn，并重编码训练输出，需要被忽略的 `runs/` 与对应 tokenizer。这个流程与普通复算分开。

刷新四个已授权 W&B history 时，安装可选 `wandb` 依赖后运行 `python scripts/fetch_wandb.py`。脚本仅在内存中从环境或项目 `.env` 读取 `WANDB_API_KEY`，只导出数值 history 与最少来源元数据。使用 `scan_history()` 避免 `history()` 的默认抽样。

## 运行项目

仓库包含购物环境、policy runtime、评测 CLI、训练适配器、Trace2Skill、配置和冻结产物。数据语料、模型 checkpoint、大型原始轨迹、本地凭据和机器缓存不进入站点包。

先阅读[环境指南](../reference/ShopSimulator/README.html)，再阅读 [runtime 与评测](../reference/docs/runtime_eval.html)。[训练指南](../reference/docs/training.html)覆盖数据准备、curriculum 校验、GPU 拓扑、slime 启动、失败分析、bare validation 与 online gate。新实验使用新输出目录，使历史产物保持可追溯。

## 检查分析契约

提取测试区分打开商品与选规格刷新，验证协议修复计模型步但不计动作，拒绝不完整 token 分母，单列技术故障，并检查配对前的任务级聚合。发布检查覆盖双语页面、站内链接与锚点、来源哈希、图表 XML、四轮 W&B 覆盖及 test 总数一致性。

## 数据包

| 证据 | 下载 |
|---|---|
| 主分析与所有技能正文 | [analysis.json](../data/analysis.json) |
| 生成组汇总 | [training_groups.csv](../data/training_groups.csv) |
| 主训练逐步指标 | [training_steps.csv](../data/training_steps.csv) |
| 四条件逐任务结果 | [test_task_outcomes.json](../data/test_task_outcomes.json) |
| Gate 版本系数 | [chunk_contributions.csv](../data/chunk_contributions.csv) |
| 行为 episode / turn 摘录 | [行为报告](behavior.html) |
| 优化器与系统 history | [W&B 诊断](optimization.html) |
| 输入哈希 | [provenance.json](../data/provenance.json) |

## 发布与复用

GitHub Pages 使用显式资源清单构建 `_site/`，凭据、原始 runs 和语料不进入包内。构建附带 checksum manifest。仓库 Pages workflow 提供部署流程，所选公开范围仍需账号与仓库支持。

ShopSimulator 保留上游来源与声明。项目不为上游材料擅自新增许可，也不把本地实验使用解释为无限制再分发授权，详见[来源说明](../reference/docs/provenance.html)与仓库现有 notices。
