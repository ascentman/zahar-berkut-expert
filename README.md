# Zahar Berkut Expert

A Retrieval-Augmented Generation (RAG) system for answering questions about Ivan Franko's novel "Захар Беркут". Includes a Streamlit chat interface, full RAG observability/logging, and an automated eval harness with LLM-as-judge and RAGAS metrics.

## Architecture

```
User question
    │
    ▼
InstrumentedRetriever
    ├── LLM generates 3-4 sub-queries (MultiQueryRetriever pattern)
    ├── Each sub-query → Qdrant similarity search (top 12 chunks)
    ├── Deduplicates chunks by content hash
    └── Records: sub_queries, similarity scores, retrieval latency
    │
    ▼
Qdrant VectorStore  ←── gemini-embedding-2 embeddings
    (local ./qdrant_db, cosine similarity, min_score=0.55)
    │
    ▼
create_stuff_documents_chain
    └── gemini-2.5-flash with page-aware system prompt
    │
    ▼
Answer + context chunks + RAGLogger entry (SQLite)
```

### Key Design Choices

| Decision | Value | Why |
|----------|-------|-----|
| Chunk size | 1000 chars / 200 overlap | Balances context density vs. retrieval precision |
| Top-k | 12 | Wider net for late-book content; dedup prevents noise |
| Multi-query | 3-4 sub-queries per question | Handles Ukrainian morphology and entity variation |
| Min similarity | 0.55 | Filters noise while retaining thematic chunks |
| LLM | gemini-2.5-flash | Fast, cheap, strong Ukrainian comprehension |
| Embeddings | gemini-embedding-2 | Best multilingual/Ukrainian coverage tested |

## Files

```
zahar-berkut-expert/
├── main.py                    # Streamlit app (thin shell, imports from chain_factory)
├── chain_factory.py           # Headless chain init (no Streamlit dep) — used by eval
├── rag_logger.py              # InstrumentedRetriever + RAGLogger (SQLite logging)
├── chunk_utilization.py       # Measures which retrieved chunks the LLM actually used
├── full-text.pdf              # Source text of "Захар Беркут"
├── qdrant_db/                 # Local Qdrant vector store (auto-created on first run)
├── rag_log.db                 # SQLite log of all queries, sub-queries, latency, chunks
└── eval/
    ├── dataset.json           # 30 ground-truth Q&A pairs (7 factual, 23 open-ended)
    ├── run_eval.py            # Eval harness CLI
    ├── metrics.py             # Scoring: factual, LLM-as-judge, RAGAS, retrieval
    ├── judge_prompt.py        # LLM-as-judge prompt (correctness/completeness/groundedness)
    └── results/               # JSON output from each eval run
```

## RAG Observability

`rag_logger.py` adds instrumentation without changing the chain interface:

- **InstrumentedRetriever** wraps the Qdrant retriever and records:
  - Sub-queries generated per question
  - Per-chunk similarity scores (stored in `doc.metadata["similarity_score"]`)
  - Retrieval wall-clock time
- **RAGLogger** persists each query to SQLite (`rag_log.db`) with full metadata
- **chunk_utilization.py** computes what fraction of retrieved chunks the LLM actually cited (content-word overlap heuristic)

> Note: Qdrant uses a local file lock. Stop Streamlit before running the eval harness — both cannot access `qdrant_db/` simultaneously.

## Eval Harness

### Dataset (`eval/dataset.json`)

30 Q&A pairs covering known RAG weak spots:

| Category | Count | Focus |
|----------|-------|-------|
| character | 7 | Character ID, motivation, relationships |
| plot | 7 | Event sequence, cause/effect, endings |
| theme | 5 | Abstract literary analysis |
| setting | 3 | Geography, time period, institutions |
| style | 2 | Literary devices, folklore elements |
| off-topic-trap | 2 | Grounding checks (must say "I don't know") |

Types: 7 factual (short exact answers) + 23 open-ended (descriptive/analytical).

### Scoring

**Factual questions** — heuristic, no LLM needed:
- Containment: is reference answer a substring of the predicted answer?
- Token overlap: Jaccard on content words (4+ chars, stop-words filtered)
- Score: `1.0` if containment, else `token_overlap`

**Open-ended questions** — LLM-as-judge (gemini-2.5-flash):
- Correctness (1-5): does the answer match the reference?
- Completeness (1-5): are all key points covered?
- Groundedness (1-5): is the answer supported by the retrieved chunks?
- Composite score: `(C + Co + G) / 15`

**RAGAS metrics** (optional, `--ragas` flag):
- `faithfulness`: are claims in the answer supported by retrieved chunks?
- `context_recall`: can the reference answer be attributed to retrieved chunks?
- Uses `ragas v0.4.3` with `LangchainLLMWrapper` + `single_turn_score(SingleTurnSample)`

**Retrieval metrics** (every run):
- avg/min/max chunk similarity scores
- chunk utilization (fraction of chunks referenced in the answer)
- sub-query on-topic rate (keyword heuristic for Ukrainian book terms)
- latency p50 / p95

### Running the Eval

```bash
# Full eval (all 30 questions)
uv run python eval/run_eval.py

# With RAGAS faithfulness + context_recall
uv run python eval/run_eval.py --ragas

# Subset by category or type
uv run python eval/run_eval.py --category theme
uv run python eval/run_eval.py --type factual

# Quick smoke test
uv run python eval/run_eval.py --max-questions 3

# Custom inter-question delay (default: 2s, or 5s with --ragas)
uv run python eval/run_eval.py --ragas --delay 10
```

Results are saved to `eval/results/eval_YYYYMMDD_HHMMSS.json` with per-question scores and an aggregate summary.

### Sample Results (Run 2 — k=12, full context)

```
Total questions:       30
Factual (7):           accuracy = 86%
Open-ended (23):       quality  = 72%
Retrieval utilization: 45%
Off-topic sub-queries: 12%
Latency p50:           4 200ms
Latency p95:           7 800ms
```

Known weak spots surfaced by the harness:
- Late-book events (Myroslava's arc, post-climax) — insufficient chunk coverage at k=12
- Abstract thematic questions — retrieval finds character names, not thematic passages
- Off-topic traps — model occasionally speculates instead of saying "I don't know"

## Setup

### Prerequisites

- Python 3.13
- `uv` package manager (`pip install uv`)
- Google AI API key ([Google AI Studio](https://aistudio.google.com/))

### Install & Run

```bash
# 1. Clone and enter the project
cd zahar-berkut-expert

# 2. Create .env with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env

# 3. Run the Streamlit app (uv installs deps automatically)
uv run streamlit run main.py

# 4. Run the eval harness (stop Streamlit first)
uv run python eval/run_eval.py --max-questions 5
```

## Tools & Libraries

| Tool | Version | Role |
|------|---------|------|
| Python | 3.13 | Runtime |
| uv | latest | Package manager |
| Streamlit | — | Chat web UI |
| LangChain | — | RAG chain orchestration |
| langchain-google-genai | — | Gemini LLM + embeddings |
| langchain-qdrant | — | Qdrant vector store integration |
| langchain-classic | — | `create_retrieval_chain`, `create_stuff_documents_chain` |
| langchain-community | — | `PyPDFLoader` |
| langchain-text-splitters | — | `RecursiveCharacterTextSplitter` |
| pypdf | — | PDF metadata + page parsing |
| Qdrant | local | Vector store (cosine similarity) |
| python-dotenv | — | `.env` loading |
| RAGAS | 0.4.3 | Faithfulness + context recall metrics |
| SQLite (stdlib) | — | RAG query log persistence |

## Screenshot

![Zahar Berkut Expert Screenshot](example.png)
