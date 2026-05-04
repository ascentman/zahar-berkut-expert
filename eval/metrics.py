import json
import re
import sys
import os

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall  # legacy API, compatible with LangchainLLMWrapper
from ragas.dataset_schema import SingleTurnSample

# Add project root so we can import chunk_utilization
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chunk_utilization import check_utilization, _content_words

from eval.judge_prompt import JUDGE_PROMPT


# --- Factual Scoring ---

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def score_factual(predicted: str, reference: str) -> dict:
    """Score a factual answer via containment + token overlap.

    Returns dict with containment (bool), token_overlap (float), score (float).
    """
    norm_pred = _normalize(predicted)
    norm_ref = _normalize(reference)

    containment = norm_ref in norm_pred

    pred_words = _content_words(predicted)
    ref_words = _content_words(reference)
    if ref_words:
        token_overlap = len(pred_words & ref_words) / len(ref_words)
    else:
        token_overlap = 1.0 if not pred_words else 0.0

    return {
        "containment": containment,
        "token_overlap": round(token_overlap, 4),
        "score": 1.0 if containment else round(token_overlap, 4),
    }


# --- LLM-as-Judge Scoring ---

def score_open_ended(
    predicted: str,
    reference: str,
    chunks: list[Document],
    llm: ChatGoogleGenerativeAI,
) -> dict:
    """Score an open-ended answer using LLM-as-judge.

    Returns dict with correctness, completeness, groundedness (1-5 each),
    explanation (str), and composite score (0-1).
    """
    context_summary = "\n".join(
        f"[p.{doc.metadata.get('page', '?')}] {doc.page_content}"
        for doc in chunks[:8]
    )

    response = llm.invoke(
        JUDGE_PROMPT.format_messages(
            reference=reference,
            predicted=predicted,
            context_summary=context_summary,
        )
    )

    try:
        # Strip markdown code fences if present
        text = response.content.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        result = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        result = {"correctness": 1, "completeness": 1, "groundedness": 1,
                  "explanation": f"Judge parse error: {response.content[:200]}"}

    composite = (result["correctness"] + result["completeness"] + result["groundedness"]) / 15.0
    result["composite_score"] = round(composite, 4)
    return result


# --- RAGAS Metrics ---

def score_ragas(
    question: str,
    predicted: str,
    reference: str,
    chunks: list[Document],
    llm: ChatGoogleGenerativeAI,
) -> dict:
    """Score using RAGAS faithfulness + context_recall.

    - faithfulness: are claims in the answer supported by retrieved chunks?
    - context_recall: can the reference answer be attributed to retrieved chunks?

    Returns dict with both scores (0-1) and a combined ragas_score.
    """
    evaluator_llm = LangchainLLMWrapper(llm)
    contexts = [doc.page_content for doc in chunks]

    faithfulness_scorer = Faithfulness(llm=evaluator_llm)
    recall_scorer = LLMContextRecall(llm=evaluator_llm)

    try:
        sample = SingleTurnSample(
            user_input=question,
            response=predicted,
            retrieved_contexts=contexts,
        )
        f_result = faithfulness_scorer.single_turn_score(sample)
        faithfulness_score = round(f_result, 4)
    except Exception as e:
        faithfulness_score = None
        print(f"  [RAGAS faithfulness error] {e}")

    try:
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=contexts,
            reference=reference,
        )
        r_result = recall_scorer.single_turn_score(sample)
        context_recall_score = round(r_result, 4)
    except Exception as e:
        context_recall_score = None
        print(f"  [RAGAS context_recall error] {e}")

    scores = [s for s in [faithfulness_score, context_recall_score] if s is not None]
    ragas_score = round(sum(scores) / len(scores), 4) if scores else None

    return {
        "faithfulness": faithfulness_score,
        "context_recall": context_recall_score,
        "ragas_score": ragas_score,
    }


# --- Retrieval Metrics ---

_BOOK_KEYWORDS = {"беркут", "тухол", "франк", "повіст", "максим", "мирослав",
                   "захар", "монгол", "громад", "боярин", "карпат"}


def _is_on_topic(sub_query: str) -> bool:
    """Check if a sub-query is about the book (keyword heuristic)."""
    lower = sub_query.lower()
    return any(kw in lower for kw in _BOOK_KEYWORDS)


def compute_retrieval_metrics(
    chunks: list[Document],
    answer: str,
    sub_queries: list[str],
) -> dict:
    """Compute retrieval quality metrics from a single query's results."""
    scores = [doc.metadata.get("similarity_score", 0) for doc in chunks]
    util_scores = check_utilization(answer, chunks)
    utilized_count = sum(1 for s in util_scores if s > 0.12)

    on_topic = [_is_on_topic(sq) for sq in sub_queries]

    return {
        "num_chunks": len(chunks),
        "avg_similarity": round(sum(scores) / len(scores), 4) if scores else 0,
        "max_similarity": round(max(scores), 4) if scores else 0,
        "min_similarity": round(min(scores), 4) if scores else 0,
        "utilization": round(utilized_count / len(chunks), 4) if chunks else 0,
        "utilized_count": utilized_count,
        "sub_queries": sub_queries,
        "sub_queries_on_topic": on_topic,
        "off_topic_count": sum(1 for t in on_topic if not t),
    }


# --- Aggregates ---

def compute_aggregates(results: list[dict]) -> dict:
    """Compute summary metrics across all eval results."""
    factual = [r for r in results if r["type"] == "factual"]
    open_ended = [r for r in results if r["type"] == "open-ended"]

    factual_accuracy = (
        sum(1 for r in factual if r["scores"]["score"] >= 0.5) / len(factual)
        if factual else 0
    )

    open_ended_quality = (
        sum(r["scores"]["composite_score"] for r in open_ended) / len(open_ended)
        if open_ended else 0
    )

    all_util = [r["retrieval"]["utilization"] for r in results]
    all_latency = sorted(r["latency_ms"] for r in results)
    off_topic_total = sum(r["retrieval"]["off_topic_count"] for r in results)
    sub_query_total = sum(len(r["retrieval"]["sub_queries"]) for r in results)

    return {
        "total_questions": len(results),
        "factual_count": len(factual),
        "open_ended_count": len(open_ended),
        "factual_accuracy": round(factual_accuracy, 4),
        "open_ended_quality": round(open_ended_quality, 4),
        "avg_retrieval_utilization": round(sum(all_util) / len(all_util), 4) if all_util else 0,
        "off_topic_rate": round(off_topic_total / sub_query_total, 4) if sub_query_total else 0,
        "latency_p50_ms": round(all_latency[len(all_latency) // 2]) if all_latency else 0,
        "latency_p95_ms": round(all_latency[int(len(all_latency) * 0.95)]) if all_latency else 0,
    }
