"""RAG Evaluation Harness for Захар Беркут Expert.

Usage:
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --max-questions 5
    uv run python eval/run_eval.py --category theme --delay 2.0
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from functools import wraps

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chain_factory import create_retrieval_chain_headless
from chunk_utilization import check_utilization
from eval.metrics import (
    compute_aggregates,
    compute_retrieval_metrics,
    score_factual,
    score_open_ended,
    score_ragas,
)


def invoke_with_retry(fn, *args, max_retries: int = 4, base_delay: float = 5.0, **kwargs):
    """Call fn(*args, **kwargs), retrying on 429 RESOURCE_EXHAUSTED with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            is_429 = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_429 and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                print(f"  [429] rate limited — retrying in {wait:.0f}s (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(wait)
            else:
                raise


def load_dataset(path: str, category: str | None = None, qtype: str | None = None) -> list[dict]:
    with open(path) as f:
        data = json.load(f)

    questions = data["questions"]
    if category:
        questions = [q for q in questions if q["category"] == category]
    if qtype:
        questions = [q for q in questions if q["type"] == qtype]
    return questions


def run_eval(args):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not set in .env")
        sys.exit(1)

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.json")
    questions = load_dataset(dataset_path, args.category, args.type)

    if args.max_questions:
        questions = questions[:args.max_questions]

    print(f"Eval: {len(questions)} questions", end="")
    if args.category:
        print(f" (category={args.category})", end="")
    if args.type:
        print(f" (type={args.type})", end="")
    print()

    # Init chain
    print("Initializing chain...")
    chain, retriever, logger = create_retrieval_chain_headless(api_key)

    # Get judge LLM (reuse same model)
    from langchain_google_genai import ChatGoogleGenerativeAI
    judge_llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", temperature=0, google_api_key=api_key
    )

    results = []

    for i, q in enumerate(questions):
        qid = q["id"]
        print(f"\n[{i+1}/{len(questions)}] {qid}: {q['question'][:60]}...", flush=True)

        # Run the chain (with 429 retry)
        t0 = time.perf_counter()
        response = invoke_with_retry(chain.invoke, {"input": q["question"]})
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = response["answer"]
        chunks = response.get("context", [])

        # Score based on type
        if q["type"] == "factual":
            scores = score_factual(answer, q["reference_answer"])
            score_display = f"score={scores['score']:.2f}"
        else:
            scores = score_open_ended(answer, q["reference_answer"], chunks, judge_llm)
            score_display = (
                f"C={scores['correctness']} Co={scores['completeness']} "
                f"G={scores['groundedness']} ({scores['composite_score']:.2f})"
            )

        # RAGAS metrics (optional, enabled via --ragas flag)
        ragas_scores = None
        if args.ragas:
            ragas_scores = invoke_with_retry(
                score_ragas,
                question=q["question"],
                predicted=answer,
                reference=q["reference_answer"],
                chunks=chunks,
                llm=judge_llm,
            )
            print(f"  ragas: faithfulness={ragas_scores['faithfulness']} "
                  f"recall={ragas_scores['context_recall']}", flush=True)

        # Retrieval metrics
        retrieval = compute_retrieval_metrics(
            chunks, answer, retriever.last_sub_queries
        )

        print(f"  {score_display} | {latency_ms:.0f}ms | "
              f"chunks={retrieval['num_chunks']} util={retrieval['utilization']:.0%}")

        result = {
            "id": qid,
            "question": q["question"],
            "reference_answer": q["reference_answer"],
            "predicted_answer": answer,
            "type": q["type"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "scores": scores,
            "retrieval": retrieval,
            "latency_ms": round(latency_ms),
        }
        if ragas_scores is not None:
            result["ragas"] = ragas_scores
        results.append(result)

        # Rate limiting
        if i < len(questions) - 1:
            time.sleep(args.delay)

    # Compute aggregates
    summary = compute_aggregates(results)

    # Build output
    run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": "gemini-2.5-flash",
            "embeddings": "gemini-embedding-2",
            "retrieval": "hybrid (vector + BM25)",
            "reranker": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            "k": 12,
            "rerank_candidates": 25,
        },
        "summary": summary,
        "results": results,
    }

    # Save
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, f"{run_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # RAGAS aggregates
    if args.ragas:
        ragas_results = [r for r in results if "ragas" in r]
        if ragas_results:
            avg_faithfulness = sum(
                r["ragas"]["faithfulness"] for r in ragas_results
                if r["ragas"]["faithfulness"] is not None
            ) / len(ragas_results)
            avg_recall = sum(
                r["ragas"]["context_recall"] for r in ragas_results
                if r["ragas"]["context_recall"] is not None
            ) / len(ragas_results)
            summary["ragas_avg_faithfulness"] = round(avg_faithfulness, 4)
            summary["ragas_avg_context_recall"] = round(avg_recall, 4)
            output["summary"] = summary

    # Print summary
    print("\n" + "=" * 60)
    print(f"EVAL SUMMARY — {run_id}")
    print("=" * 60)
    print(f"  Total questions:       {summary['total_questions']}")
    print(f"  Factual ({summary['factual_count']}):          "
          f"accuracy = {summary['factual_accuracy']:.0%}")
    print(f"  Open-ended ({summary['open_ended_count']}):       "
          f"quality  = {summary['open_ended_quality']:.0%}")
    print(f"  Retrieval utilization: {summary['avg_retrieval_utilization']:.0%}")
    print(f"  Off-topic sub-queries: {summary['off_topic_rate']:.0%}")
    print(f"  Latency p50:           {summary['latency_p50_ms']}ms")
    print(f"  Latency p95:           {summary['latency_p95_ms']}ms")
    if args.ragas and "ragas_avg_faithfulness" in summary:
        print(f"  RAGAS faithfulness:    {summary['ragas_avg_faithfulness']:.0%}")
        print(f"  RAGAS context recall:  {summary['ragas_avg_context_recall']:.0%}")
    print(f"\n  Results saved to: {output_path}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Eval Harness for Захар Беркут")
    parser.add_argument("--max-questions", type=int, help="Limit number of questions")
    parser.add_argument("--category", help="Filter by category (character, plot, theme, etc.)")
    parser.add_argument("--type", help="Filter by type (factual, open-ended)")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between questions (default: 2.0, or 5.0 with --ragas)")
    parser.add_argument("--ragas", action="store_true", help="Also run RAGAS faithfulness + context_recall metrics")
    args = parser.parse_args()

    if args.delay is None:
        args.delay = 5.0 if args.ragas else 2.0
    run_eval(args)
