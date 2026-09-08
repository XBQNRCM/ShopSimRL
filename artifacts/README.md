# 发布产物

本目录是训练与评测的**小文件摘录**，可随仓库发布。权重、完整 `traces.jsonl`、slime rollout 和 W&B 仍只在本机 `runs/`，不入库。

冷启动 \(S_0\) 与终局 \(S_T\) 分别是 `cold-start/selected_skillbank.json`（10 chunk）和 `round-003/gate/selected_skillbank.json`（8 chunk）。chunk 正文见各目录的 `selected_skill.md`。

| 轮次 | 选出 chunk 数 | analysis candidates |
|---|---:|---:|
| cold-start \(S_0\) | 10 | — |
| round-000 | 7 | 6 |
| round-001 | 5 | 6 |
| round-002 | 8 | 6 |
| round-003 \(S_T\) | 8 | 4 |

`eval/` 是本地协议 test 400 的 summary / paired comparison（coverage=1）。`eval/ranking/` 是 contribution 排序附录，对应配置在 `configs/archive/`，不是仓库入口。
