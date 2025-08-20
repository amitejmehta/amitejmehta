import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import ray
from rich.console import Console
from rich.logging import RichHandler

"""
Imagine you're a CEO who wants to build a new product. You don't write the code or design the slides yourself—
you send a message to your executive team. Each exec gets the directive and immediately springs into action:
the Head of Product starts planning, the Head of Design loops in the designers, the Data team kicks off
data cleaning, and the NLP team preps the AI models. Some teams wait for input from others; others start
immediately based on their own logic. Everyone coordinates by passing messages.

There's no central script dictating every step. No static flowchart. Just a decentralized group of experts—
each one autonomous, stateful, and reactive - driving the product forward in parallel.

This isn't a pipeline. It's not even a graph. It's an **organization of actors**.

In the **actor model**, every unit of work—whether it's writing an insight, analyzing transcripts, or rendering slides—
is an actor with its own state and message inbox. Actors react to events, make decisions independently,
and communicate with one another by sending messages. One actor's output may trigger a cascade
of work across a dozen others—just like a single CEO message can activate an entire company.

This architecture unlocks composability, concurrency, and real-world coordination. We can easily plug in
human reviewers, autonomous agents, and real-time observers. More importantly, the model is natively
scalable: actors can run on different machines, scale up and down, and be deployed across distributed systems.

In this example, we simulate a company building an AI "Insights" product using the actor model in Python—
with each department modeled as an actor, coordinating via messages, and driving parallel work toward a
shared goal.

The result isn't just a flexible architecture—it's a working metaphor for how the future of autonomous
multi-agent systems will be structured.

Introduce a standard agent to agent communication protocol and the possibilities are endless.
"""

# Setup rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, markup=True)],
)
logger = logging.getLogger("actor-system")


# ────────────────────── Common Types ────────────────────── #


@dataclass
class Message:
    sender: str
    recipient: str
    payload: Dict[str, Any]


# ────────────────────── Message Router ────────────────────── #


@ray.remote
class MessageRouter:
    def __init__(self):
        self.actors: Dict[str, Any] = {}

    def register(self, name, actor):
        self.actors[name] = actor

    def send(self, msg: Message):
        actor = self.actors.get(msg.recipient)
        if actor:
            actor.receive.remote(msg)
        else:
            logger.warning(f"[Router] Unknown recipient: {msg.recipient}")


# ────────────────────── Base Actor ────────────────────── #


class BaseActor:
    color: str = "white"

    def __init__(self, name: str, router):
        self.name = name
        self.router = router

    def log(self, message: str):
        logger.info(f"[{self.__class__.color}]{self.name}[/] {message}")

    def send_log(self, recipient: str, action: str):
        logger.info(f"[{self.__class__.color}]{self.name}[/] → [{recipient}]: {action}")


# ────────────────────── Core Business Actors ────────────────────── #


@ray.remote
class CEOActor(BaseActor):
    color: str = "red"

    def receive(self, msg: Message):
        self.log("Directive: Build 'Insights' product.")
        self.send_log("ProductLead", "Launching product development cycle")
        self.router.send.remote(
            Message(
                self.name,
                "ProductLead",
                {"directive": "Launch Insights product dev cycle"},
            )
        )


@ray.remote
class ProductLeadActor(BaseActor):
    color: str = "blue"

    def receive(self, msg: Message):
        self.log("Launching dev cycle...")
        self.send_log("DataPipelineManager", "Starting data pipeline")
        self.router.send.remote(
            Message(self.name, "DataPipelineManager", {"task": "start_pipeline"})
        )
        self.send_log("DesignLead", "Initiating slide design")
        self.router.send.remote(
            Message(self.name, "DesignLead", {"task": "start_slide_design"})
        )
        self.send_log("NLPTeamLead", "Requesting analysis module preparation")
        self.router.send.remote(
            Message(self.name, "NLPTeamLead", {"task": "prepare_analysis_module"})
        )


@ray.remote
class DataPipelineManager(BaseActor):
    color: str = "green"

    def receive(self, msg: Message):
        self.log("Data ingestion started.")
        self.send_log("TranscriptAnalyzer", "Requesting transcript analysis")
        self.router.send.remote(
            Message(self.name, "TranscriptAnalyzer", {"step": "analyze_clean_data"})
        )


@ray.remote
class TranscriptAnalyzer(BaseActor):
    color: str = "yellow"

    def receive(self, msg: Message):
        self.log("Analyzing transcripts...")
        self.send_log("InsightsWriter", "Sending analysis results")
        self.router.send.remote(
            Message(
                self.name,
                "InsightsWriter",
                {
                    "section": "analysis",
                    "content": "Customer churn drivers, trending objections, support gaps.",
                },
            )
        )


@ray.remote
class NLPTeamLead(BaseActor):
    color: str = "purple"

    def receive(self, msg: Message):
        self.log("Reviewing LLM strategy...")
        self.send_log("InsightsWriter", "Sending LLM strategy summary")
        self.router.send.remote(
            Message(
                self.name,
                "InsightsWriter",
                {
                    "section": "llm_summary",
                    "content": "Using GPT-4 to extract and cluster insight topics.",
                },
            )
        )


@ray.remote
class DesignLead(BaseActor):
    color: str = "cyan"

    def receive(self, msg: Message):
        self.log("Coordinating design team...")
        self.send_log("SlideBuilder", "Requesting mock deck rendering")
        self.router.send.remote(
            Message(self.name, "SlideBuilder", {"task": "render_mock_deck"})
        )


# ────────────────────── Additional Actors ────────────────────── #


@ray.remote
class UXResearchLead(BaseActor):
    color: str = "magenta"

    def receive(self, msg: Message):
        self.log("Extracting UX feedback...")
        self.send_log("InsightsWriter", "Sending UX research findings")
        self.router.send.remote(
            Message(
                self.name,
                "InsightsWriter",
                {
                    "section": "ux_findings",
                    "content": "Users struggle with dashboard navigation and report filtering.",
                },
            )
        )


@ray.remote
class MetricsCollector(BaseActor):
    color: str = "orange"

    def receive(self, msg: Message):
        self.log("Collecting KPIs...")
        self.send_log("InsightsWriter", "Sending metrics data")
        self.router.send.remote(
            Message(
                self.name,
                "InsightsWriter",
                {
                    "section": "metrics",
                    "content": "NPS +22, engagement +14%, churn -9%",
                },
            )
        )


@ray.remote
class LegalReview(BaseActor):
    color: str = "brown"

    def receive(self, msg: Message):
        self.log("Adding legal footer.")
        self.send_log("SlideBuilder", "Sending legal footer")
        self.router.send.remote(
            Message(
                self.name,
                "SlideBuilder",
                {
                    "task": "add_legal_footer",
                    "content": "Confidential – For internal use only",
                },
            )
        )


# ────────────────────── Writer + SlideBuilder ────────────────────── #


@ray.remote
class InsightsWriter(BaseActor):
    color: str = "white"

    def __init__(self, name, router):
        super().__init__(name, router)
        self.sections = {}

    def receive(self, msg: Message):
        section = msg.payload["section"]
        content = msg.payload["content"]
        self.sections[section] = content
        self.log(f"Section received: {section}")
        if {"analysis", "llm_summary", "ux_findings", "metrics"}.issubset(
            self.sections
        ):
            summary = "\n\n".join(
                [
                    self.sections["llm_summary"],
                    self.sections["analysis"],
                    self.sections["ux_findings"],
                    self.sections["metrics"],
                ]
            )
            self.router.send.remote(
                Message(
                    self.name,
                    "SlideBuilder",
                    {"task": "inject_summary", "content": summary},
                )
            )


@ray.remote
class SlideBuilder(BaseActor):
    color: str = "white"

    def __init__(self, name, router):
        super().__init__(name, router)
        self.slides = {}

    def receive(self, msg: Message):
        task = msg.payload["task"]
        if task == "render_mock_deck":
            self.log("Rendered visual mock deck.")
            self.slides["visual"] = "Mock visual deck layout."
        elif task == "inject_summary":
            self.log("Summary injected.")
            self.slides["content"] = msg.payload["content"]
        elif task == "add_legal_footer":
            self.log("Footer added.")
            self.slides["footer"] = msg.payload["content"]

        if {"visual", "content", "footer"} <= self.slides.keys():
            self.log(
                f"\n{self.name} Slide Deck Ready!\n---\n{self.slides['content']}\n---\n{self.slides['visual']}\n---\n{self.slides['footer']}\n"
            )


# ────────────────────── Main Launcher ────────────────────── #


def run_ray_actors():
    ray.init(ignore_reinit_error=True)

    router = MessageRouter.remote()

    def actor(cls, name):
        inst = cls.remote(name, router)
        router.register.remote(name, inst)
        return inst

    actors = {
        "CEO": actor(CEOActor, "CEO"),
        "ProductLead": actor(ProductLeadActor, "ProductLead"),
        "DataPipelineManager": actor(DataPipelineManager, "DataPipelineManager"),
        "TranscriptAnalyzer": actor(TranscriptAnalyzer, "TranscriptAnalyzer"),
        "NLPTeamLead": actor(NLPTeamLead, "NLPTeamLead"),
        "DesignLead": actor(DesignLead, "DesignLead"),
        "UXResearchLead": actor(UXResearchLead, "UXResearchLead"),
        "MetricsCollector": actor(MetricsCollector, "MetricsCollector"),
        "LegalReview": actor(LegalReview, "LegalReview"),
        "InsightsWriter": actor(InsightsWriter, "InsightsWriter"),
        "SlideBuilder": actor(SlideBuilder, "SlideBuilder"),
    }

    # Kick off the workflow
    router.send.remote(Message("system", "CEO", {}))
    router.send.remote(Message("system", "UXResearchLead", {}))
    router.send.remote(Message("system", "MetricsCollector", {}))
    router.send.remote(Message("system", "LegalReview", {}))

    # Let system run for a while
    time.sleep(6)
    ray.shutdown()


# Entry point
if __name__ == "__main__":
    run_ray_actors()
