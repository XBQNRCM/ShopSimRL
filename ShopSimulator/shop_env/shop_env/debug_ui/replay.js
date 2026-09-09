const ui = {
  runForm: document.querySelector("#runForm"),
  runPath: document.querySelector("#runPath"),
  recentRuns: document.querySelector("#recentRuns"),
  loadRun: document.querySelector("#loadRun"),
  refreshRun: document.querySelector("#refreshRun"),
  runMessage: document.querySelector("#runMessage"),
  loadDot: document.querySelector("#loadDot"),
  loadLabel: document.querySelector("#loadLabel"),
  loadMeta: document.querySelector("#loadMeta"),
  emptyState: document.querySelector("#emptyState"),
  dashboard: document.querySelector("#dashboard"),
  runName: document.querySelector("#runName"),
  runLocation: document.querySelector("#runLocation"),
  artifactStrip: document.querySelector("#artifactStrip"),
  progressLabel: document.querySelector("#progressLabel"),
  progressPercent: document.querySelector("#progressPercent"),
  progressFill: document.querySelector("#progressFill"),
  traceDiagnostics: document.querySelector("#traceDiagnostics"),
  metricGrid: document.querySelector("#metricGrid"),
  terminationChart: document.querySelector("#terminationChart"),
  errorChart: document.querySelector("#errorChart"),
  configList: document.querySelector("#configList"),
  manifestRaw: document.querySelector("#manifestRaw"),
  summaryRaw: document.querySelector("#summaryRaw"),
  filterForm: document.querySelector("#filterForm"),
  episodeSearch: document.querySelector("#episodeSearch"),
  statusFilter: document.querySelector("#statusFilter"),
  terminationFilter: document.querySelector("#terminationFilter"),
  sortEpisodes: document.querySelector("#sortEpisodes"),
  resultCount: document.querySelector("#resultCount"),
  episodeRows: document.querySelector("#episodeRows"),
  tableEmpty: document.querySelector("#tableEmpty"),
  episodeDialog: document.querySelector("#episodeDialog"),
  closeDialog: document.querySelector("#closeDialog"),
  detailTitle: document.querySelector("#detailTitle"),
  detailSubtitle: document.querySelector("#detailSubtitle"),
  detailLoading: document.querySelector("#detailLoading"),
  detailBody: document.querySelector("#detailBody"),
  detailMetrics: document.querySelector("#detailMetrics"),
  detailInstruction: document.querySelector("#detailInstruction"),
  detailInitialObservation: document.querySelector("#detailInitialObservation"),
  detailPersona: document.querySelector("#detailPersona"),
  replayTimeline: document.querySelector("#replayTimeline"),
  finalSection: document.querySelector("#finalSection"),
  conversationList: document.querySelector("#conversationList"),
  episodeRaw: document.querySelector("#episodeRaw"),
  outcomeComparison: document.querySelector("#outcomeComparison"),
};

const state = {
  runPath: "",
  payload: null,
  episodes: [],
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function isNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function formatNumber(value, digits = 2) {
  if (!isNumber(value)) return "—";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: digits }).format(value);
}

function formatPercent(value) {
  if (!isNumber(value)) return "—";
  return `${formatNumber(value * 100, 1)}%`;
}

function formatDuration(value) {
  if (!isNumber(value)) return "—";
  if (value < 1000) return `${Math.round(value)} ms`;
  if (value < 60000) return `${formatNumber(value / 1000, 1)} s`;
  return `${formatNumber(value / 60000, 1)} min`;
}

function formatBytes(value) {
  if (!isNumber(value)) return "—";
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${formatNumber(value / 1024, 1)} KB`;
  return `${formatNumber(value / 1024 ** 2, 1)} MB`;
}

function totalTokens(episode) {
  return episode.tokens?.total_tokens ?? episode.tokens?.completion_tokens ?? null;
}

function setLoading(loading, label = "正在读取实验") {
  ui.loadRun.disabled = loading;
  ui.refreshRun.disabled = loading || !state.runPath;
  ui.loadDot.classList.toggle("active", loading);
  if (loading) {
    ui.loadLabel.textContent = label;
    ui.loadMeta.textContent = "正在索引 artifact";
  }
}

function setMessage(message, kind = "normal") {
  ui.runMessage.textContent = message;
  ui.runMessage.classList.toggle("error", kind === "error");
  ui.runMessage.classList.toggle("success", kind === "success");
}

async function fetchJSON(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function appendDefinition(list, term, description) {
  list.append(element("dt", "", term), element("dd", "", description ?? "—"));
}

function renderArtifacts(artifacts) {
  ui.artifactStrip.replaceChildren();
  for (const [name, info] of Object.entries(artifacts || {})) {
    const item = element("div", `artifact-badge ${info.present ? "present" : "missing"}`);
    item.textContent = `${name} · ${info.present ? formatBytes(info.bytes) : "missing"}`;
    ui.artifactStrip.append(item);
  }
}

function renderProgress(summary, traceIndex) {
  const counts = summary.counts || {};
  const requested = counts.requested || 0;
  const recorded = counts.recorded || 0;
  const ratio = requested ? Math.min(1, recorded / requested) : 0;
  ui.progressLabel.textContent = `${recorded} / ${requested} EPISODES`;
  ui.progressPercent.textContent = `${formatNumber(ratio * 100, 1)}%`;
  ui.progressFill.style.width = `${ratio * 100}%`;
  ui.traceDiagnostics.textContent = [
    `${traceIndex.records || 0} records`,
    `${traceIndex.unique_episodes || 0} unique`,
    `${traceIndex.duplicate_records || 0} duplicates`,
    `${traceIndex.invalid_lines || 0} invalid lines`,
  ].join(" · ");
}

function metric(label, value, note, tone = "") {
  const card = element("article", `metric-card ${tone}`.trim());
  card.append(element("span", "metric-label", label), element("strong", "metric-value", value), element("small", "", note));
  return card;
}

function renderMetrics(summary) {
  const counts = summary.counts || {};
  const primary = summary.primary || {};
  const behavior = summary.behavior || {};
  const latency = summary.latency || {};
  const tokens = summary.tokens || {};
  ui.metricGrid.replaceChildren(
    metric("Recorded", counts.recorded ?? 0, `${counts.pending ?? 0} pending`),
    metric("Completed", counts.completed ?? 0, `${counts.failed ?? 0} failed`, "accent"),
    metric("Mean reward", formatNumber(primary.reward_mean, 4), `${counts.scored ?? 0} scored`, "accent"),
    metric("Success rate", formatPercent(primary.success_rate), "r_success mean", "accent"),
    metric("Avg. steps", formatNumber(behavior.steps_mean, 1), `${behavior.protocol_errors ?? 0} protocol · ${behavior.invalid_actions ?? 0} invalid`),
    metric("Avg. duration", formatDuration(latency.duration_ms_mean), `${formatNumber(tokens.total_tokens, 0)} total tokens`),
  );
}

function renderDistribution(target, values, emptyText) {
  target.replaceChildren();
  const entries = Object.entries(values || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    target.append(element("p", "distribution-empty", emptyText));
    return;
  }
  const maximum = Math.max(...entries.map(([, count]) => count), 1);
  for (const [label, count] of entries) {
    const row = element("div", "distribution-row");
    const track = element("div", "distribution-bar");
    const fill = element("i");
    fill.style.width = `${(count / maximum) * 100}%`;
    track.append(fill);
    row.append(element("label", "", label), track, element("strong", "", count));
    target.append(row);
  }
}

function renderConfig(manifest, computed) {
  const plan = manifest?.plan || {};
  const experiment = plan.experiment || manifest?.experiment_name;
  const model = plan.model || {};
  const sampling = model.sampling || model;
  const split = plan.task_split || {};
  const runtime = plan.runtime || {};
  const prompt = plan.prompt || {};
  ui.configList.replaceChildren();
  appendDefinition(ui.configList, "Experiment", typeof experiment === "string" ? experiment : experiment?.name || state.payload?.run?.name);
  appendDefinition(ui.configList, "Model", model.model || model.name || model.id || plan.model_id);
  appendDefinition(ui.configList, "Split", split.name || split.split || state.episodes[0]?.split);
  appendDefinition(ui.configList, "Sampling", `temperature ${sampling.temperature ?? "—"} · top_p ${sampling.top_p ?? "—"}`);
  appendDefinition(ui.configList, "Budget", `${sampling.max_tokens ?? "—"} tokens · ${runtime.max_steps ?? "—"} steps`);
  appendDefinition(ui.configList, "Concurrency", runtime.concurrency ?? "—");
  appendDefinition(ui.configList, "Prompt", prompt.prompt_version || prompt.id || prompt.name || "—");
  appendDefinition(ui.configList, "Protocol", runtime.action_protocol || "—");
  appendDefinition(ui.configList, "Trace schema", Object.keys(computed.schema_versions || {}).join(", ") || "—");
  ui.manifestRaw.textContent = manifest ? pretty(manifest) : "未生成 manifest.json";
  ui.summaryRaw.textContent = state.payload.artifact_summary ? pretty(state.payload.artifact_summary) : "未生成 summary.json（运行中或尚未执行 summarize）";
}

function renderTerminationOptions(episodes) {
  const selected = ui.terminationFilter.value;
  const values = [...new Set(episodes.map((episode) => episode.termination_reason).filter(Boolean))].sort();
  ui.terminationFilter.replaceChildren(element("option", "", "全部"));
  ui.terminationFilter.firstElementChild.value = "";
  for (const value of values) {
    const option = element("option", "", value);
    option.value = value;
    ui.terminationFilter.append(option);
  }
  if (values.includes(selected)) ui.terminationFilter.value = selected;
}

function episodeSearchText(episode) {
  const error = episode.error || {};
  return [
    episode.episode_id,
    episode.task_id,
    episode.sample_id,
    episode.instruction,
    episode.status,
    episode.termination_reason,
    error.type,
    error.stage,
    error.message,
  ].filter((value) => value !== undefined && value !== null).join(" ").toLowerCase();
}

function compareEpisodes(a, b) {
  const mode = ui.sortEpisodes.value;
  if (mode === "reward-asc") return (a.reward ?? Number.POSITIVE_INFINITY) - (b.reward ?? Number.POSITIVE_INFINITY);
  if (mode === "reward-desc") return (b.reward ?? Number.NEGATIVE_INFINITY) - (a.reward ?? Number.NEGATIVE_INFINITY);
  if (mode === "steps-desc") return (b.steps || 0) - (a.steps || 0);
  if (mode === "duration-desc") return (b.duration_ms || 0) - (a.duration_ms || 0);
  return (a.task_id ?? Number.MAX_SAFE_INTEGER) - (b.task_id ?? Number.MAX_SAFE_INTEGER)
    || (a.sample_id ?? 0) - (b.sample_id ?? 0)
    || String(a.episode_id).localeCompare(String(b.episode_id));
}

function statusPill(status) {
  return element("span", `status-pill ${status || "unknown"}`, status || "unknown");
}

function outcomeText(episode) {
  if (episode.termination_reason) return episode.termination_reason;
  if (episode.error) return `${episode.error.type || "Error"} @ ${episode.error.stage || "unknown"}`;
  return "—";
}

function renderEpisodeRows() {
  const query = ui.episodeSearch.value.trim().toLowerCase();
  const status = ui.statusFilter.value;
  const termination = ui.terminationFilter.value;
  const filtered = state.episodes
    .filter((episode) => !query || episodeSearchText(episode).includes(query))
    .filter((episode) => !status || episode.status === status)
    .filter((episode) => !termination || episode.termination_reason === termination)
    .sort(compareEpisodes);

  ui.episodeRows.replaceChildren();
  ui.resultCount.textContent = `${filtered.length} / ${state.episodes.length} episodes`;
  ui.tableEmpty.hidden = filtered.length > 0;
  for (const episode of filtered) {
    const row = element("tr");
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-label", `打开 task ${episode.task_id} 回放`);
    const identity = element("td", "episode-cell");
    identity.append(element("strong", "", `Task ${episode.task_id ?? "—"}`), element("small", "", `sample ${episode.sample_id ?? "—"} · ${episode.episode_id}`));
    const statusCell = element("td");
    statusCell.append(statusPill(episode.status));
    const reward = element("td", `reward-value ${!episode.reward ? "zero" : ""}`, formatNumber(episode.reward, 4));
    const outcome = element("td", "outcome-cell");
    outcome.append(element("strong", "", outcomeText(episode)));
    if (episode.error?.message) outcome.append(element("small", "", episode.error.message));
    const stepCell = element("td", "mono-cell");
    stepCell.append(element("strong", "", episode.steps ?? 0));
    const flags = [
      episode.protocol_errors ? `${episode.protocol_errors} protocol` : "",
      episode.policy_failures ? `${episode.policy_failures} policy` : "",
      episode.invalid_actions ? `${episode.invalid_actions} invalid` : "",
    ].filter(Boolean).join(" · ");
    if (flags) stepCell.append(element("small", "step-flags", flags));
    row.append(
      identity,
      statusCell,
      reward,
      outcome,
      stepCell,
      element("td", "mono-cell", formatNumber(totalTokens(episode), 0)),
      element("td", "mono-cell", formatDuration(episode.duration_ms)),
    );
    row.addEventListener("click", () => openEpisode(episode.episode_id));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openEpisode(episode.episode_id);
      }
    });
    ui.episodeRows.append(row);
  }
}

function renderRun(payload) {
  state.payload = payload;
  state.episodes = payload.episodes || [];
  ui.emptyState.hidden = true;
  ui.dashboard.hidden = false;
  ui.runName.textContent = payload.run.name;
  ui.runLocation.textContent = payload.run.path;
  renderArtifacts(payload.artifacts);
  renderProgress(payload.computed_summary, payload.trace_index);
  renderMetrics(payload.computed_summary);
  renderDistribution(ui.terminationChart, payload.computed_summary.termination_reasons, "还没有带终止原因的已评分轨迹。");
  renderDistribution(ui.errorChart, payload.computed_summary.errors, "没有 harness / infrastructure failure。");
  renderConfig(payload.manifest, payload.computed_summary);
  renderTerminationOptions(state.episodes);
  renderEpisodeRows();
}

async function loadRun(path, { refresh = false } = {}) {
  const requestedPath = path.trim();
  if (!requestedPath) return;
  setLoading(true, refresh ? "正在刷新实验" : "正在读取实验");
  setMessage("正在解析 manifest、summary，并建立 traces.jsonl 索引…");
  try {
    const payload = await fetchJSON(`/api/replay/run?path=${encodeURIComponent(requestedPath)}`);
    state.runPath = payload.run.path;
    ui.runPath.value = payload.run.path;
    renderRun(payload);
    const params = new URLSearchParams(window.location.search);
    params.set("path", payload.run.path);
    window.history.replaceState({}, "", `${window.location.pathname}?${params}`);
    window.localStorage.setItem("shopsimrl.replay.lastRun", payload.run.path);
    ui.loadLabel.textContent = payload.run.name;
    ui.loadMeta.textContent = `${payload.computed_summary.counts.recorded} episodes indexed`;
    setMessage(`已加载 ${payload.computed_summary.counts.recorded} 条唯一轨迹；点击任意 task 查看完整交互。`, "success");
  } catch (error) {
    ui.loadLabel.textContent = "加载失败";
    ui.loadMeta.textContent = "检查 run 路径或 artifact";
    setMessage(error.message, "error");
  } finally {
    setLoading(false);
  }
}

function detailMetric(label, value) {
  const card = element("div", "detail-metric");
  card.append(element("span", "", label), element("strong", "", value));
  return card;
}

function textContent(value, fallback = "—") {
  if (typeof value === "string") return value || fallback;
  if (value === undefined || value === null) return fallback;
  return pretty(value);
}

function toolCallText(toolCalls) {
  if (!Array.isArray(toolCalls) || !toolCalls.length) return "没有 tool call";
  return toolCalls.map((call) => {
    const fn = call?.function || {};
    return `${fn.name || call?.name || "unknown"}(${typeof fn.arguments === "string" ? fn.arguments : pretty(fn.arguments || call?.arguments || {})})`;
  }).join("\n");
}

function appendBadge(target, label, tone) {
  target.append(element("span", `step-badge ${tone}`, label));
}

function renderStep(step, index) {
  const model = step?.model || {};
  const environment = step?.environment || {};
  const feedback = environment?.action_feedback;
  const protocolError = step?.protocol_error || model.protocol_error;
  const policyFailure = step?.policy_failure || model.policy_failure;
  const classes = ["replay-step"];
  if (protocolError) classes.push("protocol");
  if (policyFailure) classes.push("policy");
  if (feedback?.valid === false) classes.push("invalid");
  const item = element("li", classes.join(" "));
  const header = element("header", "step-header");
  header.append(
    element("strong", "", `STEP ${step?.step ?? index + 1} · ${model.finish_reason || "model response"}`),
    element("span", "", `${formatDuration(model.latency_ms)} · ${formatNumber(model.usage?.total_tokens, 0)} tokens`),
  );

  const badges = element("div", "step-badges");
  if (protocolError) appendBadge(badges, "protocol error", "danger");
  if (policyFailure) appendBadge(badges, "policy failure", "warn");
  if (feedback?.valid === false) appendBadge(badges, "invalid action", "danger");

  const actionGrid = element("div", "step-action");
  actionGrid.append(
    element("code", "", toolCallText(model.tool_calls)),
    element("code", feedback?.valid === false ? "error-code" : "", textContent(environment.action || step?.action)),
  );

  const response = element("p", `step-feedback ${feedback?.valid === false ? "invalid" : ""}`, `Environment feedback · ${textContent(feedback || environment.observation)}`);

  const disclosures = element("div", "step-details");
  const fields = [
    ["Observation sent to model", step?.observation],
    ["Thinking / reasoning", model.reasoning_content ?? model.reasoning],
    ["Assistant content", model.content],
    ["Environment observation", environment.observation],
    ["Protocol error", protocolError],
    ["Policy failure", policyFailure],
    ["Raw model response", model],
  ];
  for (const [label, value] of fields) {
    if (value === undefined || value === null || value === "") continue;
    const details = element("details");
    details.append(element("summary", "", label), element("pre", "", textContent(value)));
    disclosures.append(details);
  }
  item.append(header);
  if (badges.childElementCount) item.append(badges);
  item.append(actionGrid, response, disclosures);
  return item;
}

function renderFinal(episode) {
  ui.finalSection.replaceChildren();
  const final = episode.final;
  const error = episode.error;
  if (final && typeof final === "object") {
    const score = element("div", "final-score");
    score.append(element("span", "section-kicker", "FINAL REWARD"), element("strong", "", formatNumber(final.reward, 4)), element("code", "", final.termination_reason || "—"));
    const result = element("div");
    result.append(element("h3", "", "任务结果"));
    const details = element("div", "final-grid");
    const finalValues = [
      ["Termination", final.termination_reason],
      ["Success", formatNumber(final.reward_detail?.r_success, 4)],
      ["Target ASIN", final.goal?.asin],
      ["Purchased ASIN", final.purchase?.asin],
    ];
    for (const [label, value] of finalValues) {
      const cell = element("div");
      cell.append(element("span", "", label), element("strong", "", value ?? "—"));
      details.append(cell);
    }
    const raw = element("details", "json-disclosure");
    raw.append(element("summary", "", "完整 final payload"), element("pre", "", pretty(final)));
    result.append(details, raw);
    ui.finalSection.append(score, result);
  } else if (error && typeof error === "object") {
    const failure = element("div", "failure-card");
    failure.append(element("span", "", `${error.type || "Error"} @ ${error.stage || "unknown"}`), element("strong", "", error.message || "轨迹生成失败"));
    const raw = element("pre", "", pretty(error));
    failure.append(raw);
    ui.finalSection.append(failure);
  } else {
    ui.finalSection.append(element("p", "distribution-empty", "该 episode 没有 final 或 error payload。"));
  }
}

function normalizedText(value) {
  return String(value ?? "").trim().toLocaleLowerCase("zh-CN");
}

function currency(value) {
  if (!isNumber(value)) return null;
  return `¥${formatNumber(value, 2)}`;
}

function productPriceText(product) {
  const selected = currency(product?.selected_price);
  if (selected) return selected;
  const pricing = product?.price;
  if (Array.isArray(pricing)) {
    const numbers = pricing.filter(isNumber);
    if (numbers.length === 1) return currency(numbers[0]);
    if (numbers.length > 1) return `${currency(Math.min(...numbers))} – ${currency(Math.max(...numbers))}`;
  }
  if (isNumber(pricing)) return currency(pricing);
  if (typeof pricing === "string" && pricing) return pricing;
  return "价格未记录";
}

function comparisonStat(label, value, tone = "") {
  const item = element("div", `comparison-stat ${tone}`.trim());
  item.append(element("span", "", label), element("strong", "", value));
  return item;
}

function optionChip(option, role, matchedPurchasedOptionKeys) {
  const selected = option?.selected === true;
  const matchesGold = selected && matchedPurchasedOptionKeys.has(normalizedText(option?.value));
  const classes = ["product-option"];
  if (selected) classes.push("selected", role === "target" ? "gold" : "purchased");
  if (role === "purchased" && selected) classes.push(matchesGold ? "matched" : "different");
  if (option?.catalog_match === false) classes.push("not-in-catalog");
  const chip = element("span", classes.join(" "));
  chip.append(element("span", "option-value", option?.value || "—"));
  const optionPrice = currency(option?.price);
  if (optionPrice) chip.append(element("small", "", optionPrice));
  if (selected) {
    chip.append(element("em", "", role === "target" ? "GOLD" : "PURCHASED"));
  }
  return chip;
}

function renderOptionAxis(axis, role, matchedPurchasedOptionKeys) {
  const block = element("div", "product-option-axis");
  block.append(element("h5", "", axis?.name || "规格"));
  const options = Array.isArray(axis?.values) ? axis.values : [];
  const ordered = [...options].sort((a, b) => Number(Boolean(b?.selected)) - Number(Boolean(a?.selected)));
  const visible = ordered.slice(0, 12);
  const overflow = ordered.slice(12);
  const values = element("div", "product-option-values");
  visible.forEach((option) => values.append(optionChip(option, role, matchedPurchasedOptionKeys)));
  if (!visible.length) values.append(element("span", "product-option empty", "没有规格值"));
  block.append(values);
  if (overflow.length) {
    const more = element("details", "more-options");
    const moreValues = element("div", "product-option-values");
    overflow.forEach((option) => moreValues.append(optionChip(option, role, matchedPurchasedOptionKeys)));
    more.append(element("summary", "", `查看其余 ${overflow.length} 个选项`), moreValues);
    block.append(more);
  }
  return block;
}

function renderProductPage(product, role, comparison) {
  const card = element("article", `product-compare-card ${role}`);
  if (!product) {
    card.append(
      element("p", "product-role", role === "target" ? "TARGET / GOLD TRUTH" : "ACTUAL PURCHASE"),
      element("h4", "", role === "target" ? "没有记录目标商品" : "Agent 未完成购买"),
      element("p", "product-empty", "该轨迹没有足够的数据来构建商品页。"),
    );
    return card;
  }

  const roleLabel = role === "target" ? "TARGET / GOLD TRUTH" : "ACTUAL PURCHASE";
  const header = element("header", "product-compare-header");
  const heading = element("div");
  heading.append(
    element("p", "product-role", roleLabel),
    element("h4", "", product.title || "未命名商品"),
  );
  const source = element(
    "span",
    `catalog-status ${product.catalog_available ? "available" : "fallback"}`,
    product.catalog_available ? "完整目录页" : "Trace 摘要",
  );
  header.append(heading, source);

  const identity = element("div", "product-identity");
  const imageWrap = element("div", "product-image-wrap");
  const imageUrl = Array.isArray(product.images) ? product.images[0] : null;
  if (imageUrl) {
    const image = element("img");
    image.src = imageUrl;
    image.alt = product.title || `${roleLabel} 商品图`;
    image.loading = "lazy";
    image.referrerPolicy = "no-referrer";
    image.addEventListener("error", () => {
      image.remove();
      imageWrap.append(element("span", "product-image-fallback", "图片不可用"));
    }, { once: true });
    imageWrap.append(image);
  } else {
    imageWrap.append(element("span", "product-image-fallback", "无商品图"));
  }
  const facts = element("div", "product-facts");
  facts.append(
    element("strong", "product-price", productPriceText(product)),
    element("code", "product-asin", `ASIN ${product.asin}`),
    element("p", "", product.brand || "品牌未记录"),
    element("p", "", product.category || "类目未记录"),
  );
  if (isNumber(product.price_upper)) facts.append(element("p", "price-limit", `任务价格上限 ¥${formatNumber(product.price_upper, 2)}`));
  identity.append(imageWrap, facts);

  const matchedTargetAttributes = new Set((comparison?.attributes?.matched || []).map(normalizedText));
  const matchedPurchasedAttributes = new Set((comparison?.attributes?.matched_purchased || []).map(normalizedText));
  const missingAttributes = new Set((comparison?.attributes?.missing || []).map(normalizedText));
  const attributes = element("div", "product-attributes");
  attributes.append(element("h5", "", "关键属性"));
  const chips = element("div", "attribute-chips");
  for (const attribute of product.attributes || []) {
    const key = normalizedText(attribute);
    let tone = "";
    if (role === "target" && matchedTargetAttributes.has(key)) tone = "matched";
    else if (role === "purchased" && matchedPurchasedAttributes.has(key)) tone = "matched";
    else if (role === "target" && missingAttributes.has(key)) tone = "missing";
    else if (role === "purchased") tone = "extra";
    chips.append(element("span", `attribute-chip ${tone}`.trim(), attribute));
  }
  if (!chips.childElementCount) chips.append(element("span", "attribute-chip empty", "没有属性记录"));
  attributes.append(chips);

  const options = element("div", "product-options");
  options.append(element("h5", "", role === "target" ? "Gold truth 规格" : "实际购买规格"));
  const matchedPurchasedOptionKeys = new Set((comparison?.options?.matched_purchased || []).map(normalizedText));
  const axes = Array.isArray(product.option_axes) ? product.option_axes : [];
  axes.forEach((axis) => options.append(renderOptionAxis(axis, role, matchedPurchasedOptionKeys)));
  if (!axes.length) options.append(element("p", "product-empty", "该商品没有规格选项。"));

  card.append(header, identity, attributes, options);
  return card;
}

function renderOutcomeComparison(comparison) {
  ui.outcomeComparison.replaceChildren();
  const header = element("header", "outcome-comparison-header");
  const copy = element("div");
  const title = element("h3", "", "目标商品 vs 实际购买");
  title.id = "outcomeComparisonTitle";
  copy.append(
    element("p", "section-kicker", "OUTCOME GAP"),
    title,
    element("p", "", "并排查看商品、属性和规格选择；绿色表示命中，红色表示目标缺失或购买差异。"),
  );
  const attributes = comparison?.attributes || {};
  const options = comparison?.options || {};
  const attributeTotal = (attributes.matched?.length || 0) + (attributes.missing?.length || 0);
  const optionTotal = (options.matched?.length || 0) + (options.missing?.length || 0);
  const stats = element("div", "comparison-stats");
  const asinValue = comparison?.asin_match === null || comparison?.asin_match === undefined
    ? "未完成购买"
    : comparison.asin_match ? "一致" : "不一致";
  stats.append(
    comparisonStat("ASIN", asinValue, comparison?.asin_match ? "matched" : "different"),
    comparisonStat("目标属性命中", `${attributes.matched?.length || 0} / ${attributeTotal}`, attributes.missing?.length ? "different" : "matched"),
    comparisonStat("Gold option 命中", `${options.matched?.length || 0} / ${optionTotal}`, options.missing?.length ? "different" : "matched"),
  );
  header.append(copy, stats);

  const legend = element("div", "comparison-legend");
  legend.append(
    element("span", "legend-gold", "Gold truth option"),
    element("span", "legend-purchased", "Agent purchased option"),
    element("span", "legend-missing", "目标未满足"),
  );
  const grid = element("div", "product-comparison-grid");
  grid.append(
    renderProductPage(comparison?.target, "target", comparison),
    renderProductPage(comparison?.purchased, "purchased", comparison),
  );
  ui.outcomeComparison.append(header, legend, grid);
}

function renderConversation(messages) {
  ui.conversationList.replaceChildren();
  if (!Array.isArray(messages) || !messages.length) {
    ui.conversationList.append(element("p", "distribution-empty", "没有 conversation messages。"));
    return;
  }
  messages.forEach((message, index) => {
    const card = element("article", `conversation-message role-${message?.role || "unknown"}`);
    const header = element("header");
    header.append(element("strong", "", `${index + 1}. ${message?.role || "unknown"}`));
    if (message?.name) header.append(element("span", "", message.name));
    const body = element("pre", "", textContent(message?.content, "(empty content)"));
    card.append(header, body);
    if (Array.isArray(message?.tool_calls) && message.tool_calls.length) {
      card.append(element("pre", "tool-call-raw", toolCallText(message.tool_calls)));
    }
    if (message?.tool_call_id) card.append(element("small", "", `tool_call_id: ${message.tool_call_id}`));
    ui.conversationList.append(card);
  });
}

function renderEpisodeDetail(payload) {
  const episode = payload.episode;
  const summary = payload.summary;
  const job = episode.job || {};
  const reset = episode.reset || {};
  ui.detailTitle.textContent = `Task ${job.task_id ?? summary.task_id ?? "—"}`;
  ui.detailSubtitle.textContent = `${episode.episode_id} · sample ${job.sample_id ?? "—"} · seed ${job.seed ?? "—"} · ${episode.schema_version || "unknown schema"}`;
  ui.detailMetrics.replaceChildren(
    detailMetric("Status", episode.status || "unknown"),
    detailMetric("Reward", formatNumber(episode.final?.reward, 4)),
    detailMetric("Termination", episode.final?.termination_reason || episode.error?.stage || "—"),
    detailMetric("Steps", Array.isArray(episode.steps) ? episode.steps.length : 0),
    detailMetric("Tokens", formatNumber(summary.tokens?.total_tokens, 0)),
    detailMetric("Duration", formatDuration(episode.duration_ms)),
  );
  ui.detailInstruction.textContent = reset.task_instruction || summary.instruction || "—";
  ui.detailInitialObservation.textContent = textContent(reset.observation);
  ui.detailPersona.textContent = textContent(reset.user_persona ?? reset.persona, "未记录 persona");
  ui.replayTimeline.replaceChildren();
  const steps = Array.isArray(episode.steps) ? episode.steps : [];
  if (steps.length) steps.forEach((step, index) => ui.replayTimeline.append(renderStep(step, index)));
  else ui.replayTimeline.append(element("li", "timeline-empty", "没有完成任何 agent step。"));
  renderFinal(episode);
  renderConversation(episode.conversation);
  ui.episodeRaw.textContent = pretty(episode);
  renderOutcomeComparison(payload.outcome_comparison || null);
  ui.detailLoading.hidden = true;
  ui.detailBody.hidden = false;
}

async function openEpisode(episodeId) {
  if (!state.runPath) return;
  ui.detailTitle.textContent = episodeId;
  ui.detailSubtitle.textContent = "正在加载完整 episode";
  ui.detailBody.hidden = true;
  ui.detailLoading.hidden = false;
  if (!ui.episodeDialog.open) ui.episodeDialog.showModal();
  try {
    const payload = await fetchJSON(`/api/replay/episode?path=${encodeURIComponent(state.runPath)}&episode_id=${encodeURIComponent(episodeId)}`);
    renderEpisodeDetail(payload);
  } catch (error) {
    ui.detailLoading.textContent = `读取失败：${error.message}`;
  }
}

async function loadRunCatalog() {
  try {
    const payload = await fetchJSON("/api/replay/runs");
    ui.recentRuns.replaceChildren();
    for (const run of payload.runs || []) {
      const option = document.createElement("option");
      option.value = run.path;
      option.label = `${run.name} · ${run.artifacts.traces ? "traces" : "no traces"}`;
      ui.recentRuns.append(option);
    }
  } catch (error) {
    setMessage(`无法读取 runs 目录：${error.message}`, "error");
  }
}

ui.runForm.addEventListener("submit", (event) => {
  event.preventDefault();
  loadRun(ui.runPath.value);
});
ui.refreshRun.addEventListener("click", () => loadRun(state.runPath, { refresh: true }));
ui.filterForm.addEventListener("submit", (event) => event.preventDefault());
ui.episodeSearch.addEventListener("input", renderEpisodeRows);
ui.statusFilter.addEventListener("change", renderEpisodeRows);
ui.terminationFilter.addEventListener("change", renderEpisodeRows);
ui.sortEpisodes.addEventListener("change", renderEpisodeRows);
ui.closeDialog.addEventListener("click", () => ui.episodeDialog.close());
ui.episodeDialog.addEventListener("click", (event) => {
  if (event.target === ui.episodeDialog) ui.episodeDialog.close();
});

async function initialize() {
  await loadRunCatalog();
  const queryPath = new URLSearchParams(window.location.search).get("path");
  const initialPath = queryPath || window.localStorage.getItem("shopsimrl.replay.lastRun") || "";
  if (initialPath) {
    ui.runPath.value = initialPath;
    await loadRun(initialPath);
  }
}

initialize();
