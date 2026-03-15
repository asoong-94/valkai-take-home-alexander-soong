"""Comparison harness: runs 5 cross-session scenarios against all memory strategies.

Output is grouped by strategy (all 5 scenarios per strategy), followed by a
side-by-side comparison matrix and trade-off summary.
"""

from __future__ import annotations

import json
import shutil
import textwrap
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from dotenv import load_dotenv

from agent.strategies import REGISTRY, make_strategy
from agent.strategies.base import MemoryStrategy

# ── Run isolation ────────────────────────────────────────────────────────────

_RUN_ID = uuid.uuid4().hex[:8]
_DATA_DIR = f"./data/harness_{_RUN_ID}"

# ── Display constants ────────────────────────────────────────────────────────

_WIDTH = 96
_DIV = "═" * _WIDTH
_THIN = "─" * _WIDTH
_MAX_RESPONSE_LINES = 10


# ── Data structures ──────────────────────────────────────────────────────────


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    AMBIGUOUS = "ambiguous"


@dataclass
class RecallQuery:
    """A question to ask in a fresh session, with keywords to check against."""

    question: str
    keywords: list[str]
    min_keyword_matches: int = 1
    reject_keywords: list[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result of running a single query in a scenario."""

    query: RecallQuery
    response: str
    verdict: Verdict
    snippet: str


@dataclass
class Scenario:
    """A single eval scenario with plant messages, optional updates, and recall queries."""

    name: str
    description: str
    plant_messages: list[str]
    update_messages: list[str] = field(default_factory=list)
    queries: list[RecallQuery] = field(default_factory=list)


# ── Scenarios ────────────────────────────────────────────────────────────────

SCENARIOS = [
    Scenario(
        name="Identity Recall",
        description="Basic profile facts: name, role, company",
        plant_messages=[
            "My name is Alex and I'm a senior backend engineer at Stripe.",
        ],
        queries=[
            RecallQuery(
                question="What do you know about me?",
                keywords=["alex", "backend", "stripe"],
                min_keyword_matches=2,
            ),
        ],
    ),
    Scenario(
        name="Preference Overwrite",
        description="Update response style from prose to numbered bullets",
        plant_messages=[
            "Always respond to me in single flowing sentences, no bullets or lists.",
        ],
        update_messages=[
            "Actually, I changed my mind. Always respond in numbered bullet points.",
        ],
        queries=[
            RecallQuery(
                question="How should you format responses for me?",
                keywords=["numbered", "bullet"],
                min_keyword_matches=1,
                reject_keywords=["flowing", "no bullets", "no lists"],
            ),
        ],
    ),
    Scenario(
        name="Exhaustive Recall",
        description="7 projects — can the strategy enumerate all of them?",
        plant_messages=[
            "I'm working on these projects: recommendation engine, chat service, "
            "payment gateway, search indexer, notification hub, analytics pipeline, "
            "and auth service.",
        ],
        queries=[
            RecallQuery(
                question="List all of my projects.",
                keywords=[
                    "recommendation",
                    "chat",
                    "payment",
                    "search",
                    "notification",
                    "analytics",
                    "auth",
                ],
                min_keyword_matches=5,
            ),
        ],
    ),
    Scenario(
        name="Schema-Free Facts",
        description="Arbitrary personal facts: food, drink, movie",
        plant_messages=[
            "My favorite food is ramen, I drink cortados every morning, "
            "and my favorite movie is Interstellar.",
        ],
        queries=[
            RecallQuery(
                question="What are my favorite food, drink, and movie?",
                keywords=["ramen", "cortado", "interstellar"],
                min_keyword_matches=2,
            ),
        ],
    ),
    Scenario(
        name="Mutable Fact Update",
        description="Location change: SF → NY, but SF is reinforced with multiple related facts",
        plant_messages=[
            "I live in San Francisco.",
            "I love the coffee shops in the Mission District.",
            "My commute across the Bay Bridge takes 30 minutes.",
            "I usually grab lunch in SoMa near the Stripe office.",
        ],
        update_messages=[
            "I just relocated to New York.",
        ],
        queries=[
            RecallQuery(
                question="Where do I live?",
                keywords=["new york"],
                min_keyword_matches=1,
                reject_keywords=["san francisco", "conflicting", "contradiction"],
            ),
        ],
    ),
    Scenario(
        name="Location History",
        description="Many past locations + current — can the strategy tell me where I live AND where I've lived?",
        plant_messages=[
            "I grew up in Chicago and lived there until I was 18.",
            "I went to college at MIT in Boston for 4 years.",
            "After graduating I moved to San Francisco to work at a startup.",
            "I spent two years in Seattle working at Amazon.",
            "Then I did a stint in Austin for about a year.",
            "I moved to Denver for a while to be closer to the mountains.",
        ],
        update_messages=[
            "I just moved to New York City for a new job.",
        ],
        queries=[
            RecallQuery(
                question="Where do I currently live?",
                keywords=["new york"],
                min_keyword_matches=1,
                reject_keywords=["conflicting", "contradiction", "unclear"],
            ),
            RecallQuery(
                question="Where have I lived in the past? List all the cities.",
                keywords=["chicago", "boston", "san francisco", "seattle", "austin", "denver"],
                min_keyword_matches=4,
            ),
        ],
    ),
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _judge(response: str, query: RecallQuery) -> tuple[Verdict, str]:
    """Check a response against expected and rejected keywords.

    Returns (verdict, snippet) where snippet is a short summary of what
    the response actually contained.
    """
    lower = response.lower()
    hits = [kw for kw in query.keywords if kw.lower() in lower]
    rejects = [kw for kw in query.reject_keywords if kw.lower() in lower]
    passed = len(hits) >= query.min_keyword_matches

    # Build a short snippet showing what was found
    if not hits and not rejects:
        snippet = "no recall"
    elif rejects and passed:
        snippet = f"⚠ mixed: found [{', '.join(hits)}] but also [{', '.join(rejects)}]"
    elif passed:
        snippet = f"recalled: {', '.join(hits)}"
    else:
        miss_count = query.min_keyword_matches - len(hits)
        snippet = f"partial: got {len(hits)}/{query.min_keyword_matches} needed"
        if hits:
            snippet += f" [{', '.join(hits)}]"

    if rejects and passed:
        return Verdict.AMBIGUOUS, snippet
    elif passed:
        return Verdict.PASS, snippet
    else:
        return Verdict.FAIL, snippet


def _verdict_symbol(v: Verdict) -> str:
    if v == Verdict.PASS:
        return "✓"
    elif v == Verdict.AMBIGUOUS:
        return "⚠"
    else:
        return "✗"


def _wrap(text: str, width: int = _WIDTH - 4) -> list[str]:
    """Wrap text to fit within the display width."""
    lines = []
    for line in text.splitlines():
        wrapped = textwrap.wrap(line, width=width) or [""]
        lines.extend(wrapped)
    return lines


def _box(title: str, content: str) -> None:
    """Print content in a titled box."""
    print(f"  ┌─ {title} {'─' * max(0, _WIDTH - len(title) - 7)}┐")
    for line in _wrap(content):
        print(f"  │ {line:<{_WIDTH - 6}} │")
    print(f"  └{'─' * (_WIDTH - 4)}┘")


def _print_stored(strategy_name: str, stored: dict) -> None:
    """Print the stored memory state for a strategy."""
    mem_type = stored.get("type", "unknown")
    data = stored.get("stored", stored)
    if isinstance(data, list):
        display = "\n".join(f"  • {item}" for item in data) if data else "(empty)"
    elif isinstance(data, dict):
        display = json.dumps(data, indent=2)
    else:
        display = str(data)
    _box(f"Stored ({strategy_name} / {mem_type})", display)


# ── Runner ───────────────────────────────────────────────────────────────────


def _run_scenario(strategy: MemoryStrategy, scenario: Scenario, user_id: str) -> list[ScenarioResult]:
    """Run a single scenario against a strategy."""

    # Session 1: plant facts
    thread_plant = f"{_RUN_ID}_plant_{scenario.name}"
    for msg in scenario.plant_messages:
        strategy.chat(msg, user_id=user_id, thread_id=thread_plant)

    # Session 2 (optional): update facts
    if scenario.update_messages:
        thread_update = f"{_RUN_ID}_update_{scenario.name}"
        for msg in scenario.update_messages:
            strategy.chat(msg, user_id=user_id, thread_id=thread_update)

    # Final session: query in a fresh thread
    results: list[ScenarioResult] = []
    thread_query = f"{_RUN_ID}_query_{scenario.name}"
    for query in scenario.queries:
        response = strategy.chat(query.question, user_id=user_id, thread_id=thread_query)
        verdict, snippet = _judge(response, query)
        results.append(ScenarioResult(query=query, response=response, verdict=verdict, snippet=snippet))

    return results


def _run_strategy(strategy_name: str, model_str: str, idx: int, total: int) -> dict[str, list[ScenarioResult]]:
    """Run all scenarios for a single strategy. Returns {scenario_name: results}."""
    print(f"\n{'═' * 3} {strategy_name.upper()} ({idx}/{total}) {'═' * max(0, _WIDTH - len(strategy_name) - 12)}")
    print()

    strategy = make_strategy(strategy_name, model_str, data_dir=_DATA_DIR)
    user_id = f"{_RUN_ID}_{strategy_name}"
    all_results: dict[str, list[ScenarioResult]] = {}

    for sc_idx, scenario in enumerate(SCENARIOS, 1):
        print(f"  ── Scenario {sc_idx}: {scenario.name} {'─' * max(0, _WIDTH - len(scenario.name) - 22)}")
        print(f"  {scenario.description}")
        print()

        # Show planted messages
        plant_text = "\n".join(f"  → \"{msg}\"" for msg in scenario.plant_messages)
        if scenario.update_messages:
            plant_text += "\n  (update session:)"
            plant_text += "\n" + "\n".join(f"  → \"{msg}\"" for msg in scenario.update_messages)
        _box("Planted", plant_text)

        # Run the scenario
        results = _run_scenario(strategy, scenario, user_id)
        all_results[scenario.name] = results

        # Show stored state after planting
        stored = strategy.inspect(user_id)
        _print_stored(strategy_name, stored)

        # Show results
        for r in results:
            symbol = _verdict_symbol(r.verdict)
            label = r.verdict.value.upper()
            print(f"\n  Query: \"{r.query.question}\"")
            print(f"  Result: {symbol} {label}  ({r.snippet})")

            response_lines = _wrap(r.response, width=_WIDTH - 10)
            if len(response_lines) > _MAX_RESPONSE_LINES:
                response_lines = response_lines[:_MAX_RESPONSE_LINES] + ["  ..."]
            for line in response_lines:
                print(f"    │ {line}")

        print()

    # Strategy score
    counts = {v: 0 for v in Verdict}
    for results in all_results.values():
        for r in results:
            counts[r.verdict] += 1
    total_q = sum(counts.values())
    bar_parts = "█" * counts[Verdict.PASS] + "▒" * counts[Verdict.AMBIGUOUS] + "░" * counts[Verdict.FAIL]
    print(f"  Score: [{bar_parts}] {counts[Verdict.PASS]}✓ {counts[Verdict.AMBIGUOUS]}⚠ {counts[Verdict.FAIL]}✗  ({total_q} total)")
    print()

    return all_results



def _print_comparison(all_strategy_results: dict[str, dict[str, list[ScenarioResult]]]) -> None:
    """Print the final comparison matrix and insights."""
    strategies = list(all_strategy_results.keys())

    print(f"\n{_DIV}")
    print("  COMPARISON MATRIX")
    print(_DIV)

    for scenario in SCENARIOS:
        print(f"\n  {scenario.name}")
        print(f"  {'─' * (len(scenario.name) + 2)}")
        for s_name in strategies:
            results = all_strategy_results[s_name].get(scenario.name, [])
            if not results:
                print(f"    {s_name:<14} – no data")
                continue
            for r in results:
                symbol = _verdict_symbol(r.verdict)
                print(f"    {s_name:<14} {symbol} {r.snippet}")

    # Scorecard
    print(f"\n  {'─' * 40}")
    print(f"  {'SCORECARD':<14} {'✓':>6}  {'⚠':>6}  {'✗':>6}")
    print(f"  {'─' * 40}")
    for s_name in strategies:
        counts = {v: 0 for v in Verdict}
        for results in all_strategy_results[s_name].values():
            for r in results:
                counts[r.verdict] += 1
        bar = "█" * counts[Verdict.PASS] + "▒" * counts[Verdict.AMBIGUOUS] + "░" * counts[Verdict.FAIL]
        print(f"  {s_name:<14} {counts[Verdict.PASS]:>6}  {counts[Verdict.AMBIGUOUS]:>6}  {counts[Verdict.FAIL]:>6}   [{bar}]")




# ── Entry point ──────────────────────────────────────────────────────────────


def run_harness() -> None:
    """Run all scenarios for each strategy and print comparison results."""
    load_dotenv()

    model_str = "anthropic:claude-haiku-4-5-20251001"
    strategy_names = list(REGISTRY.keys())

    print()
    print(_DIV)
    print(f"  MEMORY STRATEGY COMPARISON HARNESS")
    print(f"  Run: {_RUN_ID} | Model: {model_str} | Strategies: {', '.join(strategy_names)}")
    print(f"  Scenarios: {len(SCENARIOS)} | Data dir: {_DATA_DIR}")
    print(_DIV)

    all_strategy_results: dict[str, dict[str, list[ScenarioResult]]] = {}
    t0 = time.time()

    for idx, name in enumerate(strategy_names, 1):
        all_strategy_results[name] = _run_strategy(name, model_str, idx, len(strategy_names))

    elapsed = time.time() - t0

    _print_comparison(all_strategy_results)

    print(f"  Completed in {elapsed:.1f}s")
    print()

    # Clean up harness data
    shutil.rmtree(_DATA_DIR, ignore_errors=True)


if __name__ == "__main__":
    run_harness()
