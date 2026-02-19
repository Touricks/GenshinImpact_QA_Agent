"""Response Humanizer.

将 raw agent 回答润色为自然语言。Uses GRADER_MODEL (fast model).
接收 BaseChatModel，直接通过 ainvoke 调用。
"""

import logging

from .prompts import HUMANIZE_PROMPT

logger = logging.getLogger(__name__)


def _extract_text(content) -> str:
    """Extract text from AIMessage.content (handles str or list[dict] with thinking blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


class ResponseHumanizer:
    """将 raw agent 回答润色为符合人类偏好的自然语言。"""

    def __init__(self, llm):
        self.llm = llm

    async def humanize(self, question: str, raw_answer: str) -> str:
        # 短路：错误消息、空答案不走 LLM
        if not raw_answer or raw_answer.startswith("求解失败"):
            return raw_answer

        prompt = HUMANIZE_PROMPT.format(question=question, raw_answer=raw_answer)
        try:
            response = await self.llm.ainvoke(prompt)
            result = _extract_text(response.content).strip()
            # 防止 LLM 返回空内容
            if not result:
                logger.warning("Humanize returned empty, using raw answer")
                return raw_answer
            return result
        except Exception as e:
            logger.warning(f"Humanize failed: {e}, returning raw answer")
            return raw_answer
