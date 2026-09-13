import argparse
import json
import time
from pathlib import Path

import torch
from qdrant_client import models

from lab_common import MODEL_NAME, collection_name, connect, load_dataset, load_model, score_result


def summarize(rows):
    answerable = [row for row in rows if row["expected"]]
    unanswerable = [row for row in rows if not row["expected"]]
    return {
        "recall_at_k": sum(r["recall"] for r in answerable) / len(answerable) if answerable else None,
        "mrr_at_k": sum(r["rr"] for r in answerable) / len(answerable) if answerable else None,
        "mean_noise": sum(r["noise"] for r in rows) / len(rows),
        "false_accept_rate": sum(bool(r["hits"]) for r in unanswerable) / len(unanswerable) if unanswerable else None,
    }


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query")
    mode.add_argument("--eval", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k должен быть положительным")
    if args.threshold is not None and not -1 <= args.threshold <= 1:
        parser.error("--threshold должен быть от -1 до 1")

    data = load_dataset()
    name = collection_name(data["documents"])
    questions = data["questions"] if args.eval else [{"id": "manual", "query": args.query}]
    client = connect()
    rows = []
    try:
        if not client.collection_exists(name):
            raise SystemExit("Сначала выполни: uv run python index.py")
        count = client.count(name, exact=True).count
        if count != len(data["documents"]):
            raise SystemExit("Индексация неполная. Повтори: uv run python index.py")
        model = load_model()
        model.encode("query: прогрев модели", normalize_embeddings=True)

        for question in questions:
            torch.cuda.synchronize()
            started = time.perf_counter()
            vector = model.encode(f"query: {question['query']}", normalize_embeddings=True)
            torch.cuda.synchronize()
            embedding_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            points = client.query_points(
                collection_name=name, query=vector.tolist(),
                limit=args.top_k, score_threshold=args.threshold,
                search_params=models.SearchParams(exact=True), with_payload=True,
            ).points
            search_ms = (time.perf_counter() - started) * 1000
            hits = [{**point.payload, "score": point.score} for point in points]
            row = {**question, "hits": hits, "embedding_ms": embedding_ms, "search_ms": search_ms}
            print(f"\n{question['id']}: {question['query']}")
            for rank, hit in enumerate(hits, 1):
                print(f"  {rank}. {hit['id']} | {hit['score']:.4f} | {hit['title']}")
                if not args.eval:
                    print(f"     {hit['text']}")
            if not hits:
                print("  Пустая выдача: подходящих по настройкам фрагментов нет.")
            if args.eval:
                row.update(score_result(question["expected"], [hit["id"] for hit in hits]))
                print(f"  Эталон: {question['expected']}; recall={row['recall']}; noise={row['noise']}")
            rows.append(row)
    finally:
        client.close()

    summary = summarize(rows) if args.eval else None
    if summary:
        print("\nИтоги:", json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out:
        report = {
            "collection": name, "model": MODEL_NAME,
            "top_k": args.top_k, "threshold": args.threshold,
            "summary": summary, "rows": rows,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Отчёт: {args.out}")


if __name__ == "__main__":
    main()