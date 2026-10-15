"""llmgw — one client over many LLM providers."""

from .budget import BudgetExceeded
from .client import LLM, default_embed_model, default_model, get_provider, load_env, parse_model, register_provider
from .providers import FakeProvider, Provider
from .tools import Tool, as_tool, tool
from .types import (
    Attachment,
    Message,
    Request,
    Response,
    StructuredOutputError,
    ToolCall,
    ToolSpec,
    UnsupportedFeature,
    Usage,
)

__all__ = [
    "LLM", "BudgetExceeded", "default_model", "default_embed_model", "get_provider", "load_env", "parse_model", "register_provider",
    "Provider", "FakeProvider", "Tool", "tool", "as_tool",
    "Attachment", "Message", "Request", "Response", "StructuredOutputError", "ToolCall", "ToolSpec",
    "UnsupportedFeature", "Usage",
]
