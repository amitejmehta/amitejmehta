# avarra_agents/config.py
# This module handles configuration for the avarra_agents package including secrets, logging, and runtime environment.

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings
from rich.console import Console


class RichLogger(Console):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._styles = {
            "debug": "[dim]{}[/dim]",
            "info": "{}",
            "warning": "[yellow]{}[/yellow]",
            "error": "[red]{}[/red]",
            "critical": "[bold red]{}[/bold red]",
        }

    def __getattr__(self, name: str):
        if name in self._styles:
            return lambda msg: self.print(self._styles[name].format(msg))
        return super().__getattribute__(name)


class AgentSettings(BaseSettings):
    prompt_dir: Path = Field(default=Path("agents/prompts"))
    log_format: Literal["rich", "loguru"] = Field(default="loguru", alias="LOG_FORMAT")

    def configure_tracer(self):
        import logfire

        logfire.configure(scrubbing=False)
        logfire.instrument_anthropic()

    def get_logger(self):
        if self.log_format == "loguru":
            from loguru import logger

            return logger
        elif self.log_format == "rich":
            return RichLogger()
        else:
            raise ValueError(f"Invalid log format: {self.log_format}")


settings = AgentSettings()
logger = settings.get_logger()
