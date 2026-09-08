这些配置不是正式展示实验。

- ranking Top/Bottom/Random：附录对照
- `qwen35_4b_train.yaml`、`trace2skill_cold_start.example.yaml`：0830 SiliconFlow
- `trace2skill_gate_a.yaml`：历史 SiliconFlow Gate A
- `trace2skill_gate_a_local.yaml`：仅为 ranking 拟合 beta
- `trace2skill_gate_a.example.yaml`、`trace2skill_online_*.example.yaml`：未作为入口跑过的模板

正式入口在 `../`：四格 test、`qwen35_4b_val.yaml`、`qwen35_4b_val_slime.yaml`、`slime_shopsimrl.yaml`、`trace2skill_cold_start.yaml`。
