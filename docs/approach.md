# Approach & Findings

This project focuses on cross-session memory: retaining useful user information across separate conversations, not just within a single chat window. The main finding is that user memory is not one problem — it is at least three problems with different storage and update semantics: **current profile state**, **schema-free facts**, and **temporal/event history**. No single strategy handles all three well. The hybrid approach comes closest, and the evaluation suggests that a complete production design would add a third temporal/episodic layer.

## Architecture

All memory implementations share a common `MemoryStrategy` interface:

```python
class MemoryStrategy(ABC):
    def chat(self, message: str, *, user_id: str, thread_id: str) -> str: ...
    def inspect(self, user_id: str) -> dict: ...
```

- `user_id` scopes cross-session memory. `thread_id` scopes in-session conversation history.
- A registry + factory keeps the CLI, API layer, and evaluation harness independent of any specific memory implementation.

```python
REGISTRY = {
    "baseline": Baseline,
    "semantic": SemanticMemory,
    "structured": StructuredMemory,
    "hybrid": HybridMemory,
}
```

The original server was stateless (client sent full history each request). Memory strategies that manage history internally by `thread_id` required shifting session ownership to the server — the client now sends a single message and a `thread_id`.

---

## Strategies

### Baseline (control group)

In-process dict keyed by `thread_id`. No persistence. Establishes a lower bound — if baseline passes a scenario, the scenario isn't testing cross-session memory.

### Semantic memory (vector-based)

ChromaDB + sentence-transformers (`all-MiniLM-L6-v2`, local, no API key needed for embeddings).
**Pipeline:** `recall → llm → memorize`

Each turn: embed the user's message, retrieve top-k similar stored facts, prepend them as a system message, invoke the LLM, then extract new facts via `with_structured_output` and append them to ChromaDB.

**Strength:** stores any declarative fact regardless of schema — food preferences, hobbies, arbitrary details all get embedded and recalled. **Weakness:** append-only storage means old and new versions of the same fact coexist. The retrieval layer doesn't preserve latest-value semantics for mutable facts, so the LLM may surface conflicting information.

### Structured memory (fixed-schema profiles)

JSON files on disk, one per user. Pydantic `UserProfile` schema with overwrite semantics (scalars replace, lists union).
**Pipeline:** `load_profile → llm → extract_profile`

Each turn: load the profile, inject it as a system message, invoke the LLM, extract structured fields via `with_structured_output(UserProfile)`, merge and persist.

**Strength:** clean updates — when the user moves from SF to NY, the profile says NY with no ambiguity. In this prototype, all 7 projects are captured in a `projects` list and can be recalled exhaustively. **Weakness:** facts that don't map to a predefined field are silently dropped. "Favorite food is ramen" has no `UserProfile` field, so it's lost.

Note: in this prototype, `projects` lives in the structured profile for simplicity, but architecturally it's closer to episodic/enumerable data (like location history) than scalar profile state. A production design would likely move list-type fields into an event store.

### Hybrid memory (semantic + structured)

The hybrid strategy combines both stores and reduces ambiguity by separating authoritative profile state from supplementary recalled facts.
**Pipeline:** `load_profile → recall → llm → memorize → extract_profile`

Both stores persist independently per user. The profile provides current-state answers; the vector store captures everything else including historical context and schema-free facts.

**Prompt design — source of truth vs. contextual recall:**

The system message explicitly separates the two stores with different authority levels:

```
User profile (authoritative current state — these are the latest values):
Name: Alex
Location: New York City
Company: Stripe

Recalled facts (supplementary — includes historical info, personal details,
and context the profile doesn't cover):
- Grew up in Chicago until age 18
- Attended MIT in Boston for 4 years
- Moved to San Francisco after graduating to work at a startup
- Spent two years working at Amazon in Seattle
- Favorite food is ramen
```

This gives the LLM an unambiguous contract: the profile is the answer for "where do I live now?" (New York, no hedging), the recalled facts answer "where have I lived?" and "what's my favorite food?" The profile is the source of truth for anything it covers; the vector store is extended memory for everything else. Neither store tries to do the other's job.

The implemented hybrid is a practical two-layer system. The evaluation suggests a fuller production design would add a third temporal/episodic layer for mutable and enumerable history.

---

## Evaluation

The harness (`evals/harness.py`) runs 6 scenarios against all 4 strategies. Each scenario plants facts in one session, optionally updates them in a second session, then queries in a fresh third session. Responses are judged against expected keywords, rejected keywords, and minimum match thresholds with three verdicts: **PASS**, **AMBIGUOUS** (found expected but also rejected keywords), and **FAIL**.

| # | Scenario | Tests |
|---|---|---|
| 1 | Identity Recall | Basic profile facts in a new session |
| 2 | Preference Overwrite | Whether a preference update replaces the old value |
| 3 | Exhaustive Recall | Whether all 7 planted projects can be recalled |
| 4 | Schema-Free Facts | Arbitrary personal facts (food, drink, movie) |
| 5 | Mutable Fact Update | Location change SF → NY with SF reinforced by related facts |
| 6 | Location History | 6 past cities + current — can it answer both "where now?" and "where before?" |

### Results

```
SCORECARD           ✓       ⚠       ✗
baseline            1       0       6   [█░░░░░░]
semantic            4       2       1   [████▒▒░]
structured          4       1       2   [████▒░░]
hybrid              4       2       1   [████▒▒░]
```

**Baseline** — fails all scenarios that truly require cross-session memory. Preference Overwrite passes only because the base model tends to answer in bullets even without memory.

**Semantic** — flexible recall for open-ended facts (identity, food/drink/movie), but append-only storage causes ambiguity on mutable facts (location update, preference overwrite). Top-k retrieval also fragments exhaustive queries like "list all cities I've lived in."

**Structured** — precise on profile-shaped data and clean on updates (location overwrites correctly, all 7 projects recalled). Fails on schema-free facts entirely — no `UserProfile` field for "favorite food," so it's dropped.

**Hybrid** — the most production-promising design because it combines clean current-state recall with schema-free fact capture. The scorecard is similar to semantic (4/2/1), but the qualitative difference matters: hybrid answers "where do I currently live?" definitively from the profile while also being able to answer "where have I lived?" from the vector store. No other strategy can do both. Its remaining ambiguity comes from retrieval quality, not from the core separation of concerns — the retrieval layer surfaces old facts alongside the profile, and the LLM sometimes mentions historically true but currently irrelevant information. Tighter retrieval filtering (metadata, recency weighting, deduplication) would reduce this.

---

## Trade-off summary

| Dimension | Semantic | Structured | Hybrid |
|---|---|---|---|
| **Schema-free facts** | Good | Poor (drops them) | Good |
| **Mutable facts** | Poor (old + new coexist) | Good (clean overwrite) | Mostly good (profile anchors current state) |
| **Historical context** | Partial (if recalled) | None (overwritten) | Good (facts retain history) |
| **Exhaustive recall** | Poor (bounded by k) | Good (full profile) | Good (profile + wider k) |
| **Storage growth** | Unbounded | Bounded by schema | Profile bounded, facts unbounded |

The hybrid strategy exists because the two standalone approaches fail on different classes of memory problems. The profile is the source of truth for "what is X right now?" and the vector store answers "what has the user told me over time?"

The key design problem is not choosing one memory strategy, but routing each fact to the storage layer whose update and retrieval semantics match the kind of memory it represents.

---

## Future work

Three directions would materially improve the system:

1. **Metadata-aware retrieval.** Attach `category`, `created_at`, and `session_id` to each stored fact. Filter at query time to retrieve only relevant categories or recent facts, rather than flat top-k across everything. ChromaDB already supports metadata predicates — the storage layer is ready, the write path just needs to emit richer metadata.

2. **Temporal/episodic memory.** The structured profile uses pure overwrite semantics, which is correct for current state but loses history. A changelog model (shift the current value into a `history` array before overwriting) would let the agent answer "where did I used to live?" without the ambiguity of append-only vector storage. This is the missing third layer the evaluation points toward.

3. **Deduplication and canonicalization.** The append-only vector store accumulates near-duplicate facts ("Name: Alex" appears multiple times). A cosine-similarity check at write time would reduce noise and free up retrieval slots for distinct facts.
