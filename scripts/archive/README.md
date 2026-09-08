一次性或临时脚本，不是训练/评测入口。

- `recover_paired_skillbank.py`：从 0830 positive run 恢复 \(S_0\)（产物已在 `artifacts/cold-start/`）
- `build_contribution_skillbank.py`：按 Gate A 系数符号拼 control SkillBank
- `run_trace2skill_overnight.py`：Gate A 完成后串 equipped test
