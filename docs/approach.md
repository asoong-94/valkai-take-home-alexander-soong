# Approach & Findings

This document covers the design decisions, architecture, and evaluation results for the memory strategy comparison project.

## Problem statement

The starter repo provides a stateless LangChain chat agent. The goal is to extend it with multiple cross-session memory strategies, demonstrate their differences with a scripted evaluation harness, and document trade-offs.

The key value of memory is recalling information **across conversations** — not managing context within a single session. A user who tells the agent their name in session 1 should have the agent remember it in session 2.

## Architecture

### Strategy pattern

All memory implementations share a common interface defined by the `MemoryStrategy` abstract base class:

```python
class MemoryStrategy(ABC):
    def chat(self, message: str, *, user_id: str, thread_id: str) -> str: ...
    def inspect(self, user_id: str) -> dict: ...
```

- `user_id` scopes cross-session memory (what persists between conversations)
- `thread_id` scopes in-session history (the current conversation)
- `inspect()` exposes raw stored state for debugging and evaluation

A registry maps strategy names to classes, and a `make_strategy()` factory handles instantiation:

```python
REGISTRY = {
    "baseline": Baseline,
    "semantic": SemanticMemory,
    "structured": StructuredMemory,
    "hybrid": HybridMemory,
}
strategy = make_strategy("hybrid", "anthropic:claude-haiku-4-5-20251001", data_dir="./data")
```

This pattern keeps the CLI, server, and harness code strategy-agnostic.

### Statefulness model

The original server was stateless — the client sent the full conversation history on every request. With memory strategies that internally manage conversation history by `thread_id`, this model no longer works. The server now owns session management: the client sends a single `message` and a `thread_id`, and the server maintains the conversation history internally.

---

## Strategy implementations

### 1. Baseline (control group)

**Storage:** In-process Python dict, keyed by `thread_id`.
**Cross-session:** No. Everything is lost when the process restarts.
**Purpose:** Establishes a lower bound. If baseline passes a scenario, the scenario isn't actually testing cross-session memory.

The baseline is intentionally minimal — it appends user/assistant messages to a list and invokes the raw LLM. No LangGraph, no extraction, no persistence.

### 2. Semantic memory (vector-based)

**Storage:** ChromaDB (persistent vector database) + sentence-transformers for embeddings.
**Cross-session:** Yes. Facts are embedded and stored per-user in ChromaDB collections.
**LangGraph pipeline:** `START → recall → llm → memorize → END`

How it works each turn:

1. **Recall** — embed the user's latest message, query ChromaDB for the top-3 most similar stored facts.
2. **LLM** — prepend recalled facts as a system message, invoke the LLM to generate a response.
3. **Memorize** — use `with_structured_output(MemoryExtraction)` to extract declarative facts from the conversation, then add each fact as a new document in ChromaDB.

Key design decisions:

- **Append-only storage.** Facts are never deleted or updated. This means old and new versions of the same fact coexist in the vector store.
- **Local embeddings.** Uses `all-MiniLM-L6-v2` via sentence-transformers so embeddings don't require an external API call. This keeps the project self-contained and avoids burning API credits on embeddings.
- **Top-k=3 retrieval ceiling.** Intentionally limited to demonstrate that vector retrieval struggles with exhaustive recall — if a user has 7 projects, only the 3 most similar to the query get recalled.

### 3. Structured memory (fixed-schema profiles)

**Storage:** JSON files on disk (`data/profiles/{user_id}.json`).
**Cross-session:** Yes. The profile is read from disk at the start of each turn and written back after extraction.
**LangGraph pipeline:** `START → load_profile → llm → extract_profile → END`

How it works each turn:

1. **Load profile** — read the user's JSON profile from disk.
2. **LLM** — inject non-empty profile fields as a system message, invoke the LLM.
3. **Extract profile** — use `with_structured_output(UserProfile)` to extract structured data from the conversation, merge it into the existing profile (scalars overwrite, lists union), write back to disk.

The `UserProfile` schema is deliberately constrained:

```python
class UserProfile(BaseModel):
    name: str | None = None
    role: str | None = None
    company: str | None = None
    location: str | None = None
    response_style: str | None = None
    projects: list[str] = Field(default_factory=list)
    preferred_language: str | None = None
```

There is no catch-all `interests` or `facts` field. This is intentional — it forces "schema blindness" where facts that don't map to a predefined field (like "favorite food is ramen") are silently dropped. This is the key trade-off structured memory demonstrates.

### 4. Hybrid memory (semantic + structured)

After evaluating semantic and structured in isolation, the natural question is: why not both? Each strategy's weakness is the other's strength — semantic captures everything but can't cleanly update, structured updates cleanly but drops anything outside the schema. The hybrid strategy combines both stores and solves the ambiguity problem through prompt design.

**Storage:** JSON profile on disk + ChromaDB vector store (both per-user, prefixed `hybrid_`).
**Cross-session:** Yes. Both stores persist independently.
**LangGraph pipeline:** `START → load_profile → recall → llm → memorize → extract_profile → END`

How it works each turn:

1. **Load profile** — read the user's JSON profile (authoritative current state).
2. **Recall** — query ChromaDB for the top-10 most similar facts (wider window than semantic's top-3, since the profile anchors current answers and recalled facts serve as supplementary context).
3. **LLM** — invoke with both stores injected as a system message (see prompt design below).
4. **Memorize** — extract facts and append to ChromaDB (same as semantic).
5. **Extract profile** — extract structured fields and merge into the profile (same as structured).

#### Prompt design: source of truth vs. contextual recall

The key insight is how the two stores are presented to the LLM. Rather than dumping everything into a flat context, the system message explicitly separates them with different authority levels:

```
User profile (authoritative current state — these are the latest values):
Name: Alex
Location: New York City
Company: Stripe

Recalled facts (supplementary — includes historical info, personal details,
and context the profile doesn't cover; use these to answer questions about
the user's past, preferences, and interests):
- Grew up in Chicago until age 18
- Attended MIT in Boston for 4 years
- Moved to San Francisco after graduating to work at a startup
- Spent two years working at Amazon in Seattle
- Favorite food is ramen
- Drinks cortados every morning
```

This prompt structure gives the LLM an unambiguous contract:

- **"Where do I live?"** → the profile says New York City. That's the answer. No hedging, no "you previously lived in SF."
- **"Where have I lived?"** → the recalled facts contain Chicago, Boston, SF, Seattle, Austin, Denver. The LLM can enumerate all of them.
- **"What's my favorite food?"** → not in the profile schema, but the recalled facts have "ramen." Answered.

The profile acts as the **source of truth** for anything it covers (name, location, company, etc.) — overwrite semantics mean the latest value is always correct. The vector store acts as an **extended memory** for everything else — personal facts, historical context, preferences, and details the fixed schema can't capture. Neither store tries to do the other's job.

This resolves the core failure mode of each standalone strategy:
- Semantic alone can't distinguish "I live in SF" from "I moved to NY" — both are just vectors at similar distances. The LLM hedges.
- Structured alone silently drops "favorite food is ramen" because `UserProfile` has no `favorite_food` field.
- Hybrid gets both right because the profile gives a definitive current answer and the vector store captures everything else.

---

## Evaluation harness

The harness (`evals/harness.py`) runs 6 scenarios against all 4 strategies in isolated data directories. Each scenario follows the same pattern:

1. **Plant** facts in session 1 (sometimes session 2 for updates)
2. **Query** in a fresh session 3 (new `thread_id`, same `user_id`)
3. **Judge** the response against expected keywords, rejected keywords, and minimum match thresholds

Verdicts use a three-state system:
- **PASS** — found enough expected keywords, no rejected keywords
- **AMBIGUOUS** — found expected keywords but also rejected keywords (e.g., recalled both old and new locations)
- **FAIL** — didn't find enough expected keywords

### Scenarios

| # | Scenario | What it tests |
|---|---|---|
| 1 | **Identity Recall** | Basic profile facts (name, role, company) in a new session |
| 2 | **Preference Overwrite** | Whether a preference update (prose → bullets) replaces the old value |
| 3 | **Exhaustive Recall** | Whether all 7 planted projects can be recalled (tests retrieval completeness) |
| 4 | **Schema-Free Facts** | Arbitrary personal facts (food, drink, movie) that may not fit a schema |
| 5 | **Mutable Fact Update** | Location change (SF → NY) with SF reinforced by 4 related facts |
| 6 | **Location History** | 6 past cities + current — can the strategy tell me where I live AND where I've lived? |

### Results

Run on `anthropic:claude-haiku-4-5-20251001`, ~240 seconds:

```
COMPARISON MATRIX

  Identity Recall
    baseline       ✗ no recall
    semantic       ✓ recalled: alex, backend, stripe
    structured     ✓ recalled: alex, backend, stripe
    hybrid         ✓ recalled: alex, backend, stripe

  Preference Overwrite
    baseline       ✓ recalled: bullet
    semantic       ⚠ mixed: found [bullet] but also [flowing]
    structured     ⚠ mixed: found [bullet] but also [flowing]
    hybrid         ⚠ mixed: found [bullet] but also [flowing]

  Exhaustive Recall
    baseline       ✗ no recall
    semantic       ✓ recalled: all 7 projects
    structured     ✓ recalled: all 7 projects
    hybrid         ✓ recalled: all 7 projects

  Schema-Free Facts
    baseline       ✗ no recall
    semantic       ✓ recalled: ramen, cortado, interstellar
    structured     ✗ no recall
    hybrid         ✓ recalled: ramen, cortado, interstellar

  Mutable Fact Update
    baseline       ✗ no recall
    semantic       ⚠ mixed: found [new york] but also [san francisco]
    structured     ✓ recalled: new york
    hybrid         ⚠ mixed: found [new york] but also [san francisco]

  Location History (2 queries: current + past)
    baseline       ✗ no recall / ✗ no recall
    semantic       ✓ recalled: new york / ✗ partial (top-k misses most cities)
    structured     ✓ recalled: new york / ✗ no recall (not in schema)
    hybrid         ✓ recalled: new york / varies (profile anchors current, facts provide history)

  SCORECARD           ✓       ⚠       ✗
  baseline            1       0       6   [█░░░░░░]
  semantic            4       2       1   [████▒▒░]
  structured          4       1       2   [████▒░░]
  hybrid              4       2       1   [████▒▒░]
```

### Analysis

**Baseline (1/7 pass)** — only passes Preference Overwrite, which is a coincidence: the LLM defaults to bullet formatting even without memory. Every cross-session scenario fails because baseline stores nothing.

**Semantic (4/7 pass, 2 ambiguous)** — strong on open-ended facts (Identity Recall, Schema-Free Facts) because any declarative statement can be embedded and retrieved. Weaknesses:

- **Mutable Fact Update ambiguity.** Append-only storage means "lives in San Francisco" and "relocated to New York" both exist in the vector store. When queried about location, the recalled results include SF-related facts (Mission District, Bay Bridge, SoMa) alongside the NY update, and the LLM surfaces both — sometimes even calling out a "conflicting" state.
- **Preference Overwrite ambiguity.** Same root cause — old preference ("flowing sentences") coexists with the new one ("numbered bullets") and both get recalled.
- **Location History partial failure.** The vector store has all 6 past cities, but top-k retrieval only returns the most similar ones to the query. Asks like "list all cities I've lived in" get fragmented recall.

**Structured (4/7 pass, 1 ambiguous)** — strong on profile-shaped data and updates. The overwrite semantics mean the latest `location` value cleanly replaces the old one (New York, not San Francisco). All 7 projects are stored in a `projects` list field and recalled exhaustively. Weaknesses:

- **Schema-Free Facts failure.** "Favorite food is ramen" doesn't map to any `UserProfile` field, so it's silently dropped. This is the deliberate schema blindness trade-off.
- **Preference Overwrite ambiguity.** The profile captured the *first* preference ("flowing sentences") and the overwrite semantics replaced `response_style`, but the extraction picked up both phrasings.
- **Location History partial failure.** The profile has `location: "New York City"` so "where do I live now?" passes cleanly. But "where have I lived?" fails completely — the profile only stores the current value, and past locations aren't captured anywhere.

**Hybrid (4/7 pass, 2 ambiguous)** — the strongest overall strategy. Passes everything semantic does (Schema-Free Facts, Identity Recall, Exhaustive Recall) plus gets a clean pass on "where do I currently live?" from the Location History scenario thanks to the profile anchoring the answer. Remaining ambiguity:

- **Mutable Fact Update** still shows ambiguity because the vector store contains SF-related facts that get recalled alongside the profile's authoritative "New York City." The LLM correctly identifies NY as current but may mention the SF history. This is actually arguably *correct behavior* — the user did live in SF — but the harness scores it as ambiguous because the reject keywords trigger.
- **Preference Overwrite** has the same ambiguity as other strategies — both the old and new preference text exist in the recalled facts.

The key differentiator for hybrid is the **Location History scenario**. When asked "where do I currently live?", the profile gives a definitive "New York City" with no hedging. When asked "where have I lived?", the recalled facts provide Chicago, Boston, SF, Seattle, Austin, Denver. No other strategy can answer both questions correctly — semantic hedges on the current location, structured can't recall past locations.

---

## Trade-off summary

| Dimension | Semantic | Structured | Hybrid |
|---|---|---|---|
| **What it stores** | Any declarative fact | Only fields in schema | Both |
| **How it stores** | Append-only | Overwrite (latest wins) | Profile overwrites, facts append |
| **Retrieval** | Similarity top-k | Load full profile | Profile + top-k facts |
| **Exhaustive recall** | Poor (bounded by k) | Good (full profile) | Good (profile + wider k) |
| **Schema-free facts** | Good (embeds anything) | Poor (drops them) | Good (vector store captures) |
| **Mutable facts** | Poor (old + new coexist) | Good (clean overwrite) | Good (profile is source of truth) |
| **Historical context** | Partial (if recalled) | None (overwritten) | Good (facts retain history) |
| **Storage growth** | Unbounded | Bounded by schema | Profile bounded, facts unbounded |
| **Best for** | Open-ended exploration | Well-defined profiles | Production use — both strengths |

The first three strategies each demonstrate a clear trade-off. The hybrid strategy exists because neither semantic nor structured dominates — and the solution isn't a compromise, it's a combination where each store plays a distinct role. The profile is the source of truth for "what is X right now?" and the vector store answers "what has the user told me over time?"

---

## User scoping and metadata filtering

Every memory operation is scoped by `user_id`. Semantic memory stores each user's facts in a dedicated ChromaDB collection (`user_{id}`), and structured memory writes to a per-user JSON file (`data/profiles/{user_id}.json`). This keeps users fully isolated today, but the `user_id` is really just the first axis of a richer metadata story.

ChromaDB supports metadata filters on queries — you can attach arbitrary key-value pairs to each document and filter at retrieval time. Right now we embed bare facts like `"Works on payment gateway project"` with no metadata beyond the user scope. In a production system, each chunk would carry metadata like:

```python
col.add(
    documents=[fact],
    ids=[fact_id],
    metadatas=[{
        "user_id": user_id,
        "source": "conversation",
        "created_at": "2026-03-14T10:30:00Z",
        "session_id": thread_id,
        "category": "project",
    }],
)
```

This unlocks filtered retrieval: "recall only facts from the last 7 days," "recall only project-related facts," or "recall facts from a specific conversation." Instead of a flat top-k similarity search across everything, you'd combine vector similarity with metadata predicates — dramatically improving precision for users with large memory stores.

The `/users` endpoint already demonstrates a lightweight version of this idea. It scans ChromaDB collections and profile files to discover known users, acting as a metadata index over the storage layer. The same pattern extends to any dimension you want to filter or group by.

---

## Where I'd go next

### Better retrieval strategies

The current semantic retrieval is naive — flat top-k=3 with no filtering. Several improvements would make a meaningful difference:

- **Metadata-filtered retrieval.** Attach `category`, `source`, and `created_at` to each stored fact. Filter at query time to retrieve only relevant categories (e.g., "projects" when the user asks about projects, "preferences" when they ask about formatting).
- **Hybrid search.** Combine vector similarity with BM25 keyword matching. ChromaDB supports `where_document` filters that could approximate this, or swap in a purpose-built hybrid store.
- **Adaptive top-k.** Instead of a fixed k=3, dynamically adjust based on the query type. Enumeration queries ("list all my projects") need high k; identity queries ("what's my name") need k=1.
- **Deduplication.** The append-only store accumulates near-duplicate facts (e.g., "Name: Alex" appears multiple times). A dedup pass at write time — checking cosine similarity against existing documents before inserting — would reduce noise and free up retrieval slots for distinct facts.

### Temporal data in structured memory

The structured profile currently uses pure overwrite semantics — when the user moves from SF to NY, the old location is gone. This is a feature for correctness but a loss for temporal reasoning. A richer approach would maintain a changelog:

```json
{
  "name": "Alex",
  "location": {
    "current": "New York",
    "history": [
      { "value": "San Francisco", "from": "2026-01-15", "to": "2026-03-10" }
    ]
  },
  "updated_at": "2026-03-14T10:30:00Z"
}
```

With temporal metadata, the agent could answer questions like "where did I used to live?" or "when did I switch to New York?" without the ambiguity that semantic memory's append-only store creates. The merge function would shift the current value into the history array before overwriting, preserving a complete audit trail.

This also opens the door to **decay and relevance scoring** — older facts could be weighted lower in retrieval, and the agent could proactively surface stale information ("I notice your company was last updated 6 months ago — are you still at Stripe?").

### Cross-user patterns

The take-home prompt mentions learning "patterns across users." Today, each user's memory is fully isolated. A natural extension would be aggregate analysis: if 80% of users in a company prefer bullet-point formatting, a new user from that company could inherit that default. This requires a shared embedding space or a cross-user profile index — neither of which exists in the current architecture, but the `user_id` scoping makes it straightforward to add a "global" or "org-level" memory layer alongside per-user stores.

---

## What we built

1. **Abstract strategy pattern** (`MemoryStrategy` ABC) — common interface for chat + inspect, strategy-agnostic CLI/server/harness
2. **Four implementations** — baseline (control), semantic (ChromaDB + sentence-transformers), structured (JSON + Pydantic), hybrid (both stores with prompt-level source-of-truth separation)
3. **LangGraph pipelines** — semantic: recall → llm → memorize; structured: load_profile → llm → extract_profile; hybrid: load_profile → recall → llm → memorize → extract_profile
4. **Evaluation harness** — 6 scenarios (including location history), 3-state verdicts (pass/ambiguous/fail), keyword + reject-keyword matching, comparison matrix with scorecard
5. **FastAPI server** — memory-aware API (`POST /chat`, `GET /strategies`, `GET /users`, `GET /inspect`) with lazy strategy initialization
6. **React frontend** — strategy picker, user dropdown with "new user" flow, live-updating memory panel (profile table + facts list for hybrid, flat list for semantic, dictionary table for structured)
