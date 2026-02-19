"""LLM Factory — 根据 model name 自动选择 provider 并创建 LangChain ChatModel。"""

import logging

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

GEMINI_PREFIXES = ("gemini-", "models/gemini-")
CLAUDE_PREFIXES = ("claude-",)
OPENAI_PREFIXES = ("gpt-", "o1-", "o3-", "o4-")

_cache: dict[tuple, BaseChatModel] = {}


def create_chat_model(model_name: str, **kwargs) -> BaseChatModel:
    """根据 model name 自动选择 provider 并创建 LangChain ChatModel。

    Provider 路由规则：
      - "gemini-*" / "models/gemini-*"  → ChatGoogleGenerativeAI
      - "claude-*"                       → ChatAnthropic
      - "gpt-*" / "o1-*" / "o3-*"      → ChatOpenAI

    kwargs 直接透传给对应 ChatModel 构造器（如 thinking_level, thinking_budget 等）。
    实例缓存: 相同 (model_name, kwargs) 返回同一实例。
    """
    key = (model_name, tuple(sorted(kwargs.items())))
    if key in _cache:
        logger.debug(f"Using cached model: {model_name}")
        return _cache[key]

    name = model_name.lower()

    if any(name.startswith(p) for p in GEMINI_PREFIXES):
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info(f"Creating Gemini model: {model_name}")
        model = ChatGoogleGenerativeAI(model=model_name, **kwargs)
    elif any(name.startswith(p) for p in CLAUDE_PREFIXES):
        from langchain_anthropic import ChatAnthropic
        logger.info(f"Creating Claude model: {model_name}")
        model = ChatAnthropic(model=model_name, **kwargs)
    elif any(name.startswith(p) for p in OPENAI_PREFIXES):
        from langchain_openai import ChatOpenAI
        logger.info(f"Creating OpenAI model: {model_name}")
        model = ChatOpenAI(model=model_name, **kwargs)
    else:
        raise ValueError(
            f"Unknown model provider for '{model_name}'. "
            f"Expected prefix: gemini-*, claude-*, gpt-*, o1-*, o3-*"
        )

    _cache[key] = model
    return model
