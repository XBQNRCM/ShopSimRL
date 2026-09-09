"use strict";

const ui = {
  healthDot: document.querySelector("#healthDot"),
  healthLabel: document.querySelector("#healthLabel"),
  healthMeta: document.querySelector("#healthMeta"),
  resetForm: document.querySelector("#resetForm"),
  taskId: document.querySelector("#taskId"),
  taskMode: document.querySelector("#taskMode"),
  interactionMode: document.querySelector("#interactionMode"),
  startButton: document.querySelector("#startButton"),
  releaseButton: document.querySelector("#releaseButton"),
  sessionBadge: document.querySelector("#sessionBadge"),
  formMessage: document.querySelector("#formMessage"),
  turnCounter: document.querySelector("#turnCounter"),
  contextGrid: document.querySelector("#contextGrid"),
  visibleInstruction: document.querySelector("#visibleInstruction"),
  personaOutput: document.querySelector("#personaOutput"),
  observationOutput: document.querySelector("#observationOutput"),
  stateOutput: document.querySelector("#stateOutput"),
  rawOutput: document.querySelector("#rawOutput"),
  copyObservation: document.querySelector("#copyObservation"),
  actionForm: document.querySelector("#actionForm"),
  actionInput: document.querySelector("#actionInput"),
  sendButton: document.querySelector("#sendButton"),
  availableActions: document.querySelector("#availableActions"),
  rewardPanel: document.querySelector("#rewardPanel"),
  primaryReward: document.querySelector("#primaryReward"),
  purchaseMatch: document.querySelector("#purchaseMatch"),
  rewardGrid: document.querySelector("#rewardGrid"),
  rewardRaw: document.querySelector("#rewardRaw"),
  traceList: document.querySelector("#traceList"),
  clearTrace: document.querySelector("#clearTrace"),
};

const session = {
  envIdx: null,
  leaseId: null,
  turn: 0,
  done: false,
  busy: false,
  latestResult: null,
  lastAction: "",
};

function pretty(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function formatValue(value) {
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  return String(value);
}

function setMessage(message = "", isError = false) {
  ui.formMessage.textContent = message;
  ui.formMessage.classList.toggle("error", isError);
}

function setBusy(busy) {
  session.busy = busy;
  ui.startButton.disabled = busy;
  ui.releaseButton.disabled = busy || session.envIdx === null;
  const canAct = !busy && session.envIdx !== null && !session.done;
  ui.actionInput.disabled = !canAct;
  ui.sendButton.disabled = !canAct;
  for (const button of ui.availableActions.querySelectorAll("button")) {
    button.disabled = !canAct;
  }
}

async function callApi(payload, { keepalive = false } = {}) {
  const response = await fetch("/api/shop_agent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    keepalive,
  });
  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`环境返回了非 JSON 响应（HTTP ${response.status}）`);
  }
  const result = body?.result;
  if (!response.ok || !result || result.error) {
    throw new Error(result?.error || `环境请求失败（HTTP ${response.status}）`);
  }
  return result;
}

async function refreshHealth() {
  try {
    const response = await fetch("/healthz", { cache: "no-store" });
    const health = await response.json();
    if (!response.ok || !health.ready) throw new Error("环境尚未就绪");
    ui.healthDot.className = "status-dot ready";
    ui.healthLabel.textContent = "环境已就绪";
    ui.healthMeta.textContent = `${health.environment_version} · ${health.observation_version} · ${health.free_slots}/${health.environment_slots} slots · ${health.primary_reward}`;
  } catch (error) {
    ui.healthDot.className = "status-dot error";
    ui.healthLabel.textContent = "环境不可用";
    ui.healthMeta.textContent = error.message;
  }
}

function renderAvailableActions(result) {
  ui.availableActions.replaceChildren();
  const actions = Array.isArray(result?.observation_state?.actions)
    ? result.observation_state.actions
    : [];
  if (!actions.length || result.done) {
    const empty = document.createElement("span");
    empty.className = "empty-hint";
    empty.textContent = result.done ? "Episode 已终止" : "当前页面没有 click 动作";
    ui.availableActions.append(empty);
    return;
  }
  for (const value of actions) {
    const button = document.createElement("button");
    button.className = "action-chip";
    button.type = "button";
    button.textContent = `click[${value}]`;
    button.title = "使用环境给出的精确值执行";
    button.addEventListener("click", () => executeAction(`click[${value}]`));
    ui.availableActions.append(button);
  }
}

function renderResult(result) {
  session.latestResult = result;
  session.done = Boolean(result.done);
  ui.observationOutput.textContent = result.observation || "（环境没有返回 observation）";
  ui.stateOutput.textContent = pretty(result.observation_state);
  ui.rawOutput.textContent = pretty(result);
  ui.turnCounter.textContent = session.turn ? `TURN ${session.turn}` : "RESET";
  renderAvailableActions(result);
  if (result.done) renderReward(result);
  setBusy(session.busy);
}

function renderContext(result) {
  ui.contextGrid.hidden = false;
  ui.visibleInstruction.textContent = result.task_instruction || "—";
  ui.personaOutput.textContent = result.user_persona ? pretty(result.user_persona) : "当前任务未提供 Persona";
}

function rewardMetric(label, value) {
  const box = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = formatValue(value);
  box.append(term, description);
  return box;
}

function renderReward(result) {
  const detail = result.reward_detail || {};
  ui.rewardPanel.hidden = false;
  ui.primaryReward.textContent = formatValue(result.reward);
  ui.rewardGrid.replaceChildren();
  const metrics = [
    ["termination", result.termination_reason],
    ["r_loose", detail.r_loose],
    ["r_strict", detail.r_strict],
    ["r_success", detail.r_success],
    ["r_type", detail.r_type],
    ["r_att", detail.r_att],
    ["r_option", detail.r_option],
    ["r_price", detail.r_price],
    ["option matches", detail.num_option_matches],
    ["attribute matches", detail.num_attr_matches],
    ["price verifiable", detail.price_verifiable],
    ["target ASIN", detail.target_asin_match],
  ];
  const shownKeys = new Set([
    "termination_reason",
    "r_loose",
    "r_strict",
    "r_success",
    "r_type",
    "r_att",
    "r_option",
    "r_price",
    "num_option_matches",
    "num_attr_matches",
    "price_verifiable",
    "target_asin_match",
  ]);
  for (const [key, value] of Object.entries(detail)) {
    if (!shownKeys.has(key) && (value === null || ["string", "number", "boolean"].includes(typeof value))) {
      metrics.push([key, value]);
    }
  }
  for (const [label, value] of metrics) {
    ui.rewardGrid.append(rewardMetric(label, value));
  }

  const purchasedAsin = result.purchase?.asin ?? "—";
  const goalAsin = result.goal?.asin ?? "—";
  const match = String(purchasedAsin) === String(goalAsin);
  ui.purchaseMatch.replaceChildren();
  const summary = document.createElement("strong");
  summary.textContent = match ? "ASIN MATCH" : "ASIN DIFFERENT";
  const details = document.createTextNode(`\nPurchased  ${purchasedAsin}\nGoal       ${goalAsin}`);
  ui.purchaseMatch.append(summary, details);
  ui.rewardRaw.textContent = pretty(detail);
}

function resetTrace() {
  ui.traceList.replaceChildren();
}

function appendTrace({ label, action, result }) {
  if (ui.traceList.querySelector(".trace-empty")) ui.traceList.replaceChildren();
  const item = document.createElement("li");
  item.className = `trace-item${result.done ? " done" : ""}`;
  const header = document.createElement("header");
  const step = document.createElement("span");
  step.textContent = label;
  const status = document.createElement("span");
  status.textContent = result.done ? `DONE · ${formatValue(result.reward)}` : result.observation_state?.page_type || "OK";
  header.append(step, status);
  const command = document.createElement("code");
  command.textContent = action;
  const detail = document.createElement("div");
  detail.className = "trace-result";
  const actionCount = result.observation_state?.actions?.length ?? 0;
  detail.textContent = result.done
    ? result.termination_reason || "terminal"
    : `${actionCount} click actions available`;
  item.append(header, command, detail);
  ui.traceList.append(item);
}

async function releaseLease({ quiet = false, keepalive = false } = {}) {
  if (session.envIdx === null || !session.leaseId) return;
  const payload = {
    action: "release_one",
    env_idx: session.envIdx,
    lease_id: session.leaseId,
  };
  await callApi(payload, { keepalive });
  session.envIdx = null;
  session.leaseId = null;
  ui.sessionBadge.textContent = session.done ? "已终止 · 已释放" : "已释放";
  ui.sessionBadge.classList.remove("active");
  if (!quiet) setMessage("环境槽位已释放；当前页面结果仍保留。", false);
  setBusy(session.busy);
  refreshHealth();
}

async function startEpisode(event) {
  event.preventDefault();
  setBusy(true);
  setMessage("正在创建环境会话…");
  try {
    if (session.envIdx !== null) await releaseLease({ quiet: true });
    const taskId = Number(ui.taskId.value);
    if (!Number.isInteger(taskId) || taskId < 0) throw new Error("Task ID 必须是非负整数");
    const payload = {
      action: "reset",
      idx: taskId,
      if_persona: ui.taskMode.value === "persona",
      interaction_mode: ui.interactionMode.value,
    };
    const result = await callApi(payload);
    session.envIdx = result.env_idx;
    session.leaseId = result.lease_id;
    session.turn = 0;
    session.done = false;
    session.lastAction = "";
    resetTrace();
    ui.rewardPanel.hidden = true;
    renderContext(result);
    renderResult(result);
    appendTrace({ label: "RESET", action: `reset(task=${taskId})`, result });
    ui.sessionBadge.textContent = `slot ${result.env_idx} · task ${taskId}`;
    ui.sessionBadge.classList.add("active");
    setMessage(`已启动 ${result.environment_version || "ShopSimulator"}`);
    ui.actionInput.focus();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    setBusy(false);
    refreshHealth();
  }
}

async function executeAction(action) {
  const normalized = action.trim();
  if (!normalized || session.envIdx === null || session.done || session.busy) return;
  setBusy(true);
  setMessage("正在执行动作…");
  try {
    const result = await callApi({
      action: "interact",
      env_idx: session.envIdx,
      lease_id: session.leaseId,
      response: normalized,
    });
    session.turn += 1;
    session.lastAction = normalized;
    renderResult(result);
    appendTrace({ label: `TURN ${session.turn}`, action: normalized, result });
    ui.actionInput.value = "";
    const feedback = result.action_feedback || result.observation_state?.last_action;
    if (result.done) {
      setMessage(`Episode 已终止：${result.termination_reason || "done"}`);
    } else if (feedback && feedback.valid === false) {
      setMessage(`动作无效：${feedback.message || feedback.reason}`, true);
    } else {
      setMessage("动作执行完成");
    }
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    setBusy(false);
  }
}

ui.resetForm.addEventListener("submit", startEpisode);
ui.actionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  executeAction(ui.actionInput.value);
});
ui.actionInput.addEventListener("keydown", (event) => {
  if (event.key === "ArrowUp" && !ui.actionInput.value && session.lastAction) {
    event.preventDefault();
    ui.actionInput.value = session.lastAction;
  }
});
ui.releaseButton.addEventListener("click", async () => {
  setBusy(true);
  try {
    await releaseLease();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    setBusy(false);
  }
});
ui.copyObservation.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(ui.observationOutput.textContent);
    ui.copyObservation.textContent = "已复制";
    window.setTimeout(() => { ui.copyObservation.textContent = "复制"; }, 1200);
  } catch {
    setMessage("浏览器未允许复制，请手动选择文本。", true);
  }
});
ui.clearTrace.addEventListener("click", () => {
  resetTrace();
  const empty = document.createElement("li");
  empty.className = "trace-empty";
  empty.textContent = "轨迹显示已清空；环境会话不受影响。";
  ui.traceList.append(empty);
});
window.addEventListener("beforeunload", () => {
  if (session.envIdx === null || !session.leaseId) return;
  fetch("/api/shop_agent", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action: "release_one",
      env_idx: session.envIdx,
      lease_id: session.leaseId,
    }),
    keepalive: true,
  });
});

refreshHealth();
