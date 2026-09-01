"""Versioned prompt assembly kept separate from the agent state machine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

from .schemas import Skill, fingerprint


DEFAULT_SYSTEM_PROMPT = """你正在进行一次网上购物模拟，目标是从商品库中选购最符合需求的商品。
我会提供目标商品以及用户个人文档。商品库中存在大量同类商品，你必须综合分析用户需求和当前页面信息，最终购买最符合用户要求的商品。

每一轮必须且只能调用一个当前提供的函数工具：
1. search：仅在该工具可用时调用，根据当前需求填写简洁、有效的检索关键词。
2. click：从工具参数列出的当前可点击值中逐字选择一个，不得自行编造。

规则说明：
1. 在调用 click 购买前，必须完成当前商品的全部规格轴。
2. 通过搜索、比较、规格选择和购买完成任务，不要随意购买仅部分满足需求的商品。
3. 工具结果会返回新的页面观察；根据最新观察继续调用下一步工具。
4. 每轮只返回一个标准 tool call，不要在最终响应中输出 Thought、Action、JSON、XML 或额外自然语言。
5. 请在模型的 thinking 阶段完成分析，最终答案仅使用提供的函数工具。
"""


@dataclass(frozen=True)
class ShoppingPromptBuilder:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt_version: str = "shopsim-shopping-agent-tools-v2"
    skill_renderer_version: str = "plain-skill-context-v1"

    def identity(self) -> dict[str, Any]:
        return {
            "prompt_version": self.prompt_version,
            "system_prompt_sha256": fingerprint(self.system_prompt),
            "skill_renderer_version": self.skill_renderer_version,
        }

    def system_message(
        self, reset: dict[str, Any], skills: Sequence[Skill]
    ) -> str:
        sections = [self.system_prompt.rstrip()]
        persona = reset.get("user_persona")
        if isinstance(persona, dict) and persona:
            sections.append(
                "用户个人文档：\n"
                + json.dumps(persona, ensure_ascii=False, sort_keys=True)
            )
        if skills:
            sections.append(self.render_skills(skills))
        return "\n\n".join(sections)

    def render_skills(self, skills: Sequence[Skill]) -> str:
        lines = [
            "[可选策略提示]",
            "这些提示只提供通用策略；仍需依据当前任务和页面独立核验。",
        ]
        for skill in skills:
            lines.extend(
                [
                    f"<skill id={json.dumps(skill.skill_id, ensure_ascii=False)} "
                    f"version={json.dumps(skill.version, ensure_ascii=False)}>",
                    skill.content.strip(),
                    "</skill>",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def initial_observation(reset: dict[str, Any]) -> str:
        return (
            f"购物任务：\n{reset['task_instruction']}\n\n"
            f"{reset['observation']}"
        )
