import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict

from rich.console import Console
from rich.logging import RichHandler

"""
Imagine you're a CEO who wants to build a new product. You don't write the code or design the slides yourself—
you send a message to your executive team. Each exec gets the directive and immediately springs into action:
the Head of Product starts planning, the Head of Design loops in the designers, the Data team kicks off
data cleaning, and the NLP team preps the AI models. Some teams wait for input from others; others start
immediately based on their own logic. Everyone coordinates by passing messages.

There's no central script dictating every step. No static flowchart. Just a decentralized group of experts—
each one autonomous, stateful, and reactive—driving the product forward in parallel.

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


# Messaging + base actor classes
@dataclass
class Message:
    sender: str
    recipient: str
    payload: Dict[str, Any]


class Actor:
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        self.name = name
        self.inbox = inbox
        self.router = router
        self.color = "white"  # Default color, will be overridden by subclasses

    async def run(self):
        while True:
            msg: Message = await self.inbox.get()
            await self.receive(msg)

    async def receive(self, message: Message):
        raise NotImplementedError

    def log(self, message: str):
        logger.info(f"[{self.color}]{self.name}[/] {message}")

    def send_log(self, recipient: str, action: str):
        logger.info(f"[{self.color}]{self.name}[/] → [{recipient}]: {action}")


class MessageRouter:
    def __init__(self):
        self.queues: Dict[str, asyncio.Queue] = {}

    def register(self, actor_name: str, queue: asyncio.Queue):
        self.queues[actor_name] = queue

    async def send(self, message: Message):
        recipient_queue = self.queues.get(message.recipient)
        if recipient_queue:
            await recipient_queue.put(message)
        else:
            logger.warning(f"[Router] Unknown recipient: {message.recipient}")


# ──────────────────────── Core Business Actors ───────────────────────── #


class CEOActor(Actor):
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        super().__init__(name, inbox, router)
        self.color = "red"

    async def receive(self, message: Message):
        self.log("Directive: Build 'Insights' product.")
        await asyncio.sleep(0.5)
        self.send_log("ProductLead", "Launching product development cycle")
        await self.router.send(
            Message(
                self.name,
                "ProductLead",
                {"directive": "Launch Insights product dev cycle"},
            )
        )


class ProductLeadActor(Actor):
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        super().__init__(name, inbox, router)
        self.color = "blue"

    async def receive(self, message: Message):
        self.log("Launching dev cycle for Insights...")
        await asyncio.sleep(0.3)
        self.send_log("DataPipelineManager", "Starting data pipeline")
        await self.router.send(
            Message(self.name, "DataPipelineManager", {"task": "start_pipeline"})
        )
        self.send_log("DesignLead", "Initiating slide design")
        await self.router.send(
            Message(self.name, "DesignLead", {"task": "start_slide_design"})
        )
        self.send_log("NLPTeamLead", "Requesting analysis module preparation")
        await self.router.send(
            Message(self.name, "NLPTeamLead", {"task": "prepare_analysis_module"})
        )


class DataPipelineManager(Actor):
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        super().__init__(name, inbox, router)
        self.color = "green"

    async def receive(self, message: Message):
        self.log("Kicking off data ingestion + cleaning.")
        await asyncio.sleep(1)
        self.send_log("TranscriptAnalyzer", "Requesting transcript analysis")
        await self.router.send(
            Message(self.name, "TranscriptAnalyzer", {"step": "analyze_clean_data"})
        )


class TranscriptAnalyzerActor(Actor):
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        super().__init__(name, inbox, router)
        self.color = "yellow"

    async def receive(self, message: Message):
        self.log("Analyzing cleaned transcripts...")
        await asyncio.sleep(1.2)
        insights = "Customer churn drivers, trending objections, support gaps."
        self.send_log("InsightsWriter", "Sending analysis results")
        await self.router.send(
            Message(
                self.name,
                "InsightsWriter",
                {"section": "analysis", "content": insights},
            )
        )


class NLPTeamLead(Actor):
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        super().__init__(name, inbox, router)
        self.color = "magenta"

    async def receive(self, message: Message):
        self.log("Reviewing current LLM strategy...")
        await asyncio.sleep(0.7)
        self.send_log("InsightsWriter", "Sending LLM strategy summary")
        await self.router.send(
            Message(
                self.name,
                "InsightsWriter",
                {
                    "section": "llm_summary",
                    "content": "Using GPT-4 to extract and cluster insight topics.",
                },
            )
        )


class DesignLead(Actor):
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        super().__init__(name, inbox, router)
        self.color = "cyan"

    async def receive(self, message: Message):
        self.log("Coordinating with visual designers...")
        await asyncio.sleep(0.9)
        self.send_log("SlideBuilder", "Requesting mock deck rendering")
        await self.router.send(
            Message(self.name, "SlideBuilder", {"task": "render_mock_deck"})
        )


# ──────────────────────── Add-on Insight Sources ───────────────────────── #


class UXResearchLead(Actor):
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        super().__init__(name, inbox, router)
        self.color = "bright_blue"

    async def receive(self, message: Message):
        self.log("Gathering UX pain points from feedback logs...")
        await asyncio.sleep(0.8)
        insights = "Users struggle with dashboard navigation and report filtering."
        self.send_log("InsightsWriter", "Sending UX research findings")
        await self.router.send(
            Message(
                self.name,
                "InsightsWriter",
                {"section": "ux_findings", "content": insights},
            )
        )


class MetricsCollector(Actor):
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        super().__init__(name, inbox, router)
        self.color = "bright_green"

    async def receive(self, message: Message):
        self.log("Aggregating KPIs for insights product...")
        await asyncio.sleep(1.1)
        kpis = "NPS +22, engagement +14%, churn -9%"
        self.send_log("InsightsWriter", "Sending metrics data")
        await self.router.send(
            Message(
                self.name, "InsightsWriter", {"section": "metrics", "content": kpis}
            )
        )


class LegalReviewActor(Actor):
    def __init__(self, name: str, inbox: asyncio.Queue, router: "MessageRouter"):
        super().__init__(name, inbox, router)
        self.color = "bright_magenta"

    async def receive(self, message: Message):
        self.log("Reviewing legal compliance for report distribution...")
        await asyncio.sleep(0.6)
        self.send_log("SlideBuilder", "Sending legal footer")
        await self.router.send(
            Message(
                self.name,
                "SlideBuilder",
                {
                    "task": "add_legal_footer",
                    "content": "Confidential – For internal use only",
                },
            )
        )


# ──────────────────────── Writer + Slide Builder ───────────────────────── #


class ExtendedInsightsWriter(Actor):
    def __init__(self, name, inbox, router):
        super().__init__(name, inbox, router)
        self.parts = {}
        self.color = "bright_yellow"

    async def receive(self, message: Message):
        section = message.payload["section"]
        content = message.payload["content"]
        self.parts[section] = content
        self.log(f"Integrated section: {section}")
        if {"analysis", "llm_summary", "ux_findings", "metrics"}.issubset(self.parts):
            await asyncio.sleep(0.4)
            summary = "\n\n".join(
                [
                    self.parts["llm_summary"],
                    self.parts["analysis"],
                    self.parts["ux_findings"],
                    self.parts["metrics"],
                ]
            )
            await self.router.send(
                Message(
                    self.name,
                    "SlideBuilder",
                    {"task": "inject_summary", "content": summary},
                )
            )


class ExtendedSlideBuilder(Actor):
    def __init__(self, name, inbox, router):
        super().__init__(name, inbox, router)
        self.slides = {}
        self.color = "bright_cyan"

    async def receive(self, message: Message):
        task = message.payload["task"]
        if task == "render_mock_deck":
            self.log("Created visual mockups.")
            self.slides["visual"] = (
                "Mock slide deck with visual layout and placeholders."
            )
        elif task == "inject_summary":
            self.slides["content"] = message.payload["content"]
            self.log("Injected full insight summary into slides.")
        elif task == "add_legal_footer":
            self.slides["footer"] = message.payload["content"]
            self.log("Added legal compliance footer.")

        if {"visual", "content", "footer"} <= self.slides.keys():
            await asyncio.sleep(0.4)
            final = (
                f"[bold cyan]Final Slide Deck[/bold cyan]\n"
                f"---\n{self.slides['content']}\n---\n{self.slides['visual']}\n"
                f"---\n[italic]{self.slides['footer']}[/italic]"
            )
            console.rule("[bold green]Slide Deck Complete")
            console.print(final)
            console.rule("[bold green]End")


# ─────────────────────────── Main Orchestration ────────────────────────── #


async def run_richer_actor_system():
    router = MessageRouter()
    actors = {
        "CEO": CEOActor("CEO", asyncio.Queue(), router),
        "ProductLead": ProductLeadActor("ProductLead", asyncio.Queue(), router),
        "DataPipelineManager": DataPipelineManager(
            "DataPipelineManager", asyncio.Queue(), router
        ),
        "DesignLead": DesignLead("DesignLead", asyncio.Queue(), router),
        "NLPTeamLead": NLPTeamLead("NLPTeamLead", asyncio.Queue(), router),
        "TranscriptAnalyzer": TranscriptAnalyzerActor(
            "TranscriptAnalyzer", asyncio.Queue(), router
        ),
        "InsightsWriter": ExtendedInsightsWriter(
            "InsightsWriter", asyncio.Queue(), router
        ),
        "SlideBuilder": ExtendedSlideBuilder("SlideBuilder", asyncio.Queue(), router),
        "UXResearchLead": UXResearchLead("UXResearchLead", asyncio.Queue(), router),
        "MetricsCollector": MetricsCollector(
            "MetricsCollector", asyncio.Queue(), router
        ),
        "LegalReview": LegalReviewActor("LegalReview", asyncio.Queue(), router),
    }

    for name, actor in actors.items():
        router.register(name, actor.inbox)

    tasks = [asyncio.create_task(actor.run()) for actor in actors.values()]

    # Fire off multiple entry points
    await router.send(Message("system", "CEO", {}))
    await router.send(Message("system", "UXResearchLead", {}))
    await router.send(Message("system", "MetricsCollector", {}))
    await router.send(Message("system", "LegalReview", {}))

    await asyncio.sleep(8)
    for task in tasks:
        task.cancel()


# # Run in notebook or script
# import nest_asyncio
# nest_asyncio.apply()
asyncio.run(run_richer_actor_system())
