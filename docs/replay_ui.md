# 实验回放盘面

ShopSimRL Replay 是环境服务内置的只读实验浏览器。它直接读取项目根目录 `runs/` 下的标准 artifact，不生成第二份数据库，也不改变采样、评测或 `summarize` 的结果。

## 使用

先按原方式启动 ShopSimulator：

```powershell
cd ShopSimulator\shop_env
.\start.ps1
```

浏览器打开 `http://127.0.0.1:5700/replay-ui`，输入以下任一形式：

```text
qwen35-4b-test-0830
runs\qwen35-4b-test-0830
C:\project\ShopSimRL\runs\qwen35-4b-test-0830
```

为了避免把调试 API 变成任意文件浏览器，绝对路径也必须位于项目 `runs/` 内。需要把 artifact 放在其他根目录时，启动环境前设置 `SHOPSIM_RUNS_ROOT`。

页面支持：

- 查看 manifest、落盘 summary 和由当前 traces 实时计算的完成度、奖励、成功率、step、token、延迟、终止原因与错误分布；
- 按 task、episode、指令或错误文本检索，按状态和终止原因筛选，并按 reward、step、耗时排序；
- 点击 episode 查看任务、persona、初始 observation，以及每一步 thinking、标准 tool call、canonical action、环境反馈、协议纠错和完整 conversation；
- 在回放末尾并排对比 target ASIN 与 purchased ASIN 的商品页，突出 gold truth option、实际购买 option，以及目标属性和规格的命中与缺失；
- 在实验仍在运行时点击“刷新”，查看新追加的 JSONL 记录。

Agent View 和 Experiment Replay 共用同一套 Flask 服务与视觉导航，可在 `/debug-ui` 和 `/replay-ui` 之间切换。

## 数据流与边界

```text
manifest.json ── jobs/config ───────────────┐
summary.json  ── 已落盘汇总（原样展示）──────┤─> run dashboard
traces.jsonl  ── 轻量 episode 索引/实时汇总 ┘
                         │
                         └─ byte offset ─> 单 episode 完整回放
```

总览请求扫描 `traces.jsonl`，但只返回每条 episode 的轻量摘要，不把全部 observation 和 conversation 发给浏览器。点击 task 时，服务通过索引中的 byte offset 只读取对应 JSONL 记录。索引以文件大小和修改时间为缓存签名，因此 append 后刷新会自动重建。

断点续跑可能让同一 `episode_id` 在 JSONL 中出现多次。回放盘面与 runner 的完成态语义一致：索引保留最后一条记录，并单独报告总记录数、唯一 episode 数和重复数。进程恰好在 append 中断时留下的空行或损坏尾行会被计入 `invalid_lines` 并跳过；文件后续写完整后，下一次刷新会重新识别。

进度的 `requested` 来自 `manifest.plan.jobs`，`recorded` 来自唯一 episode，二者之差显示为 pending。Dashboard 的可比指标始终由当前唯一 trace 实时重算；`summary.json` 作为实验产物原样保留在“落盘 summary”中，便于判断它是否过期或尚未生成。

## 实现位置

- `ShopSimulator/shop_env/shop_env/replay.py`：路径约束、JSON/JSONL 解析、offset 索引和实时汇总；
- `ShopSimulator/shop_env/shop_env/pack_api.py`：页面和三个只读 `/api/replay/*` 接口；
- `ShopSimulator/shop_env/shop_env/debug_ui/replay.html|css|js`：总览、筛选表格和单 episode 时间线；
- `ShopSimulator/shop_env/tests/test_replay.py`：重复记录、损坏尾行、实时指标和路径边界测试。

当前设计针对项目的 `shopsimrl-run-manifest-v1` 与 `shopsimrl-episode-v4`，解析器对缺失的可选字段使用空值展示，并在盘面上列出实际检测到的 trace schema。若未来对 trace 做不兼容升级，应在 `replay.py` 中增加显式版本分支，而不是在前端猜测字段。
