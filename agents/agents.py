from pathlib import Path

from .llm import LLM
from .tools import edit_file, execute_bash, read_file


def claude_code() -> LLM:
    """Create a Claude Code agent."""
    llm = LLM(
        system_prompt=Path("CLAUDE.md").read_text(),
        tools=[read_file, edit_file, execute_bash],
    )
    return llm


AGENT_MAP = {"code": claude_code}
