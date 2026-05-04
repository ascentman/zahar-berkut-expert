import json
import sqlite3
import time
from datetime import datetime, timezone

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from pydantic import Field
from typing_extensions import override


# --- SQLite Logging Backend ---

class RAGLogger:
    def __init__(self, db_path: str = "./observability.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_query TEXT NOT NULL,
                sub_queries_json TEXT,
                answer TEXT,
                retrieval_time_ms REAL,
                total_time_ms REAL,
                num_chunks INTEGER
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                content TEXT,
                page INTEGER,
                similarity_score REAL,
                utilization_score REAL,
                was_utilized INTEGER,
                FOREIGN KEY (query_id) REFERENCES queries(id)
            );
        """)
        self.conn.commit()

    def log_query(self, user_query: str, sub_queries: list[str], answer: str,
                  retrieval_time_ms: float, total_time_ms: float, num_chunks: int) -> int:
        cursor = self.conn.execute(
            """INSERT INTO queries (timestamp, user_query, sub_queries_json, answer,
                                    retrieval_time_ms, total_time_ms, num_chunks)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(timezone.utc).isoformat(), user_query,
             json.dumps(sub_queries, ensure_ascii=False), answer,
             retrieval_time_ms, total_time_ms, num_chunks)
        )
        self.conn.commit()
        return cursor.lastrowid

    def log_chunks(self, query_id: int, chunks: list[Document],
                   utilization_scores: list[float], threshold: float = 0.05):
        rows = []
        for doc, util_score in zip(chunks, utilization_scores):
            rows.append((
                query_id,
                doc.page_content[:500],
                doc.metadata.get("page"),
                doc.metadata.get("similarity_score"),
                util_score,
                1 if util_score > threshold else 0,
            ))
        self.conn.executemany(
            """INSERT INTO chunks (query_id, content, page, similarity_score,
                                   utilization_score, was_utilized)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows
        )
        self.conn.commit()

    def get_stats(self) -> dict:
        row = self.conn.execute("""
            SELECT COUNT(*) as total_queries,
                   AVG(retrieval_time_ms) as avg_retrieval_ms,
                   AVG(total_time_ms) as avg_total_ms,
                   AVG(num_chunks) as avg_chunks
            FROM queries
        """).fetchone()
        util_row = self.conn.execute("""
            SELECT AVG(CAST(was_utilized AS REAL)) as avg_utilization
            FROM chunks
        """).fetchone()
        return {
            "total_queries": row["total_queries"],
            "avg_retrieval_ms": row["avg_retrieval_ms"] or 0,
            "avg_total_ms": row["avg_total_ms"] or 0,
            "avg_chunks": row["avg_chunks"] or 0,
            "avg_utilization": util_row["avg_utilization"] or 0,
        }

    def get_recent_queries(self, limit: int = 10) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# --- Instrumented Retriever ---

class _LineListOutputParser(BaseOutputParser[list[str]]):
    @override
    def parse(self, text: str) -> list[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))


_MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a retrieval assistant for the Ukrainian novel "Захар Беркут" by Ivan Franko.
Generate 2 alternative versions of the user's question to find relevant passages in the book.
Rules:
- All questions must be about the book's content (characters, plot, themes, settings)
- Use Ukrainian language
- Do NOT generate questions about real-world people or events outside the book
- Focus on different aspects: character traits, plot events, relationships, themes
Provide the alternative questions separated by newlines.
Original question: {question}""",
)


class InstrumentedRetriever(BaseRetriever):
    """Retriever that captures similarity scores and timing metadata.

    Replicates MultiQueryRetriever behavior but calls
    similarity_search_with_score() to capture cosine scores.
    """

    vector_store: QdrantVectorStore
    llm: BaseLanguageModel
    k: int = 8
    min_score: float = 0.55
    include_original: bool = True

    # Stored after each retrieval for the caller to read
    last_sub_queries: list[str] = Field(default_factory=list)
    last_retrieval_time_ms: float = 0.0

    class Config:
        arbitrary_types_allowed = True

    def _generate_sub_queries(self, query: str) -> list[str]:
        chain = _MULTI_QUERY_PROMPT | self.llm | _LineListOutputParser()
        sub_queries = chain.invoke({"question": query})
        if self.include_original:
            sub_queries.insert(0, query)
        return sub_queries

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        t0 = time.perf_counter()

        # Generate sub-queries
        sub_queries = self._generate_sub_queries(query)
        self.last_sub_queries = sub_queries

        # Search with scores for each sub-query, deduplicate keeping best score
        best_by_content: dict[str, tuple[Document, float, str]] = {}
        for sq in sub_queries:
            results = self.vector_store.similarity_search_with_score(sq, k=self.k)
            for doc, score in results:
                key = (doc.page_content, doc.metadata.get("page"))
                if key not in best_by_content or score > best_by_content[key][1]:
                    best_by_content[key] = (doc, score, sq)

        # Filter by minimum score, sort descending, take top-k
        sorted_results = sorted(best_by_content.values(), key=lambda x: x[1], reverse=True)
        filtered = [(d, s, q) for d, s, q in sorted_results if s >= self.min_score]
        top_results = filtered[:self.k] if filtered else sorted_results[:3]

        # Inject metadata
        docs = []
        for doc, score, source_query in top_results:
            doc.metadata["similarity_score"] = round(score, 4)
            doc.metadata["sub_query_source"] = source_query
            docs.append(doc)

        self.last_retrieval_time_ms = (time.perf_counter() - t0) * 1000
        return docs
