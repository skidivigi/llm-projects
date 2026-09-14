import argparse
import hashlib
import json
import time
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from lab_common import MODEL_NAME, ROOT, connect, load_model


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def load_data():
    path = ROOT / "data/chunk_dataset.json"
    if not path.exists():
        raise SystemExit("Сначала выполни: uv run python prepare_chunks.py")
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = {doc["id"]: doc for doc in data["documents"]}
    if not docs or len(docs) != len(data["documents"]):
        raise ValueError("Нужны документы с уникальными id")
    if not data["questions"]:
        raise ValueError("Нет контрольных вопросов")
    for question in data["questions"]:
        if set(question["expected"]) != {e["doc_id"] for e in question["evidence"]}:
            raise ValueError(f"{question['id']}: expected и evidence не согласованы")
        for evidence in question["evidence"]:
            text = docs[evidence["doc_id"]]["text"]
            start, end = evidence["start"], evidence["end"]
            if not 0 <= start < end <= len(text) or text[start:end] != evidence["quote"]:
                raise ValueError("Разметка устарела. Повтори prepare_chunks.py")
    return data


def make_chunks(docs, tokenizer, size, overlap, max_input=512):
    if not 0 <= overlap < size:
        raise ValueError("Нужно 0 <= overlap < size")
    if not tokenizer.is_fast:
        raise ValueError("Для координат текста нужен fast tokenizer")
    chunks = []
    for doc in docs:
        text = doc["text"]
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if not offsets:
            raise ValueError(f"Пустой документ: {doc['id']}")
        for number, token_start in enumerate(range(0, len(offsets), size - overlap)):
            token_end = min(token_start + size, len(offsets))
            start = offsets[token_start][0] if token_start else 0
            end = offsets[token_end][0] if token_end < len(offsets) else len(text)
            chunk_text = text[start:end]
            input_tokens = len(tokenizer.encode("passage: " + chunk_text, truncation=False))
            if not chunk_text.strip() or input_tokens > max_input:
                raise ValueError(f"{doc['id']}: чанк пуст или превышает лимит {max_input}; уменьши size")
            chunks.append({
                "chunk_id": f"{doc['id']}:{number}", "doc_id": doc["id"],
                "title": doc["title"], "text": chunk_text,
                "start": start, "end": end,
                "token_start": token_start, "token_end": token_end,
                "body_tokens": len(tokenizer.encode(chunk_text, add_special_tokens=False)),
                "input_tokens": input_tokens,
            })
            if token_end == len(offsets):
                break
    return chunks


def merge_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def measure(question, hits):
    found_docs = {hit["doc_id"] for hit in hits}
    expected = set(question["expected"])
    ranges = {
        doc_id: merge_ranges([(h["start"], h["end"]) for h in hits if h["doc_id"] == doc_id])
        for doc_id in found_docs
    }
    covered = [
        any(start <= e["start"] and end >= e["end"] for start, end in ranges.get(e["doc_id"], []))
        for e in question["evidence"]
    ]
    total_chars = sum(h["end"] - h["start"] for h in hits)
    unique_chars = sum(end - start for spans in ranges.values() for start, end in spans)
    return {
        "doc_recall": len(expected & found_docs) / len(expected) if expected else None,
        "evidence_recall": sum(covered) / len(covered) if covered else None,
        "covered": covered,
        "duplicate_share": (total_chars - unique_chars) / total_chars if total_chars else 0.0,
        "context_tokens": sum(h["body_tokens"] for h in hits),
    }


def summarize(rows):
    positive = [r for r in rows if r["expected"]]
    negative = [r for r in rows if not r["expected"]]
    return {
        "doc_recall": sum(r["doc_recall"] for r in positive) / len(positive) if positive else None,
        "evidence_recall": sum(r["evidence_recall"] for r in positive) / len(positive) if positive else None,
        "false_accept_rate": sum(bool(r["hits"]) for r in negative) / len(negative) if negative else None,
        "mean_duplicate_share": sum(r["duplicate_share"] for r in rows) / len(rows),
        "mean_context_tokens": sum(r["context_tokens"] for r in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["index", "eval", "inspect"])
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--question", help="ID одного контрольного вопроса, например q11")
    parser.add_argument("--doc", default="backup_prod", help="Документ для inspect")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.size < 1 or not 0 <= args.overlap < args.size or args.top_k < 1:
        parser.error("Нужны size > overlap >= 0 и top-k > 0")
    if args.threshold is not None and not -1 <= args.threshold <= 1:
        parser.error("Порог должен быть от -1 до 1")
    data = load_data()
    if args.action == "inspect":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        chunks = make_chunks(data["documents"], tokenizer, args.size, args.overlap)
        selected = [c for c in chunks if c["doc_id"] == args.doc]
        if not selected:
            parser.error("Неизвестный --doc")
        for chunk in selected:
            print(f"\n{chunk['chunk_id']} | исходные токены [{chunk['token_start']}, {chunk['token_end']})")
            print(f"Вход модели: {chunk['input_tokens']} токенов; символы [{chunk['start']}, {chunk['end']})")
            print(chunk["text"])
        return

    import torch
    from qdrant_client import models

    model = load_model()
    chunks = make_chunks(data["documents"], model.tokenizer, args.size, args.overlap, model.max_seq_length)
    config = {
        "model": MODEL_NAME, "distance": "Cosine", "chunker": "offset_windows_v1",
        "size": args.size, "overlap": args.overlap, "max_input": model.max_seq_length,
    }
    name = "rag_lab_step03_" + digest({"config": config, "chunks": chunks})[:12]
    print(f"Коллекция: {name}; документов: {len(data['documents'])}; чанков: {len(chunks)}")
    client = connect()
    try:
        if args.action == "index":
            vectors = model.encode(
                ["passage: " + c["text"] for c in chunks],
                batch_size=32, normalize_embeddings=True, show_progress_bar=True,
            )
            if not client.collection_exists(name):
                client.create_collection(
                    collection_name=name,
                    vectors_config=models.VectorParams(size=int(vectors.shape[1]), distance=models.Distance.COSINE),
                )
            client.upsert(
                collection_name=name, wait=True,
                points=[models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, c["chunk_id"])), vector=v.tolist(), payload=c,
                ) for c, v in zip(chunks, vectors)],
            )
            print(f"Записано точек: {client.count(name, exact=True).count}")
            return

        if not client.collection_exists(name) or client.count(name, exact=True).count != len(chunks):
            raise SystemExit("Сначала выполни index с такими же --size и --overlap")
        questions = [q for q in data["questions"] if not args.question or q["id"] == args.question]
        if not questions:
            parser.error("Неизвестный --question")
        model.encode("query: прогрев модели", normalize_embeddings=True)
        rows = []
        for question in questions:
            torch.cuda.synchronize()
            started = time.perf_counter()
            vector = model.encode("query: " + question["query"], normalize_embeddings=True)
            torch.cuda.synchronize()
            embedding_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            points = client.query_points(
                collection_name=name, query=vector.tolist(), limit=args.top_k,
                score_threshold=args.threshold, with_payload=True,
                search_params=models.SearchParams(exact=True),
            ).points
            search_ms = (time.perf_counter() - started) * 1000
            hits = [{**p.payload, "score": p.score} for p in points]
            metrics = measure(question, hits)
            rows.append({**question, "hits": hits, **metrics, "embedding_ms": embedding_ms, "search_ms": search_ms})
            print(f"\n{question['id']}: {question['query']}")
            for rank, hit in enumerate(hits, 1):
                print(f"  {rank}. {hit['chunk_id']} | score={hit['score']:.4f}")
                if args.question:
                    print(hit["text"])
            print(f"  DocRecall={metrics['doc_recall']}; EvidenceRecall={metrics['evidence_recall']}")
            if args.question:
                for evidence, covered in zip(question["evidence"], metrics["covered"]):
                    print(f"  Покрыт={covered}: {evidence['doc_id']} | {evidence['quote']}")
        summary = summarize(rows)
        print("\nИтоги:", json.dumps(summary, ensure_ascii=False, indent=2))
        if args.out:
            report = {
                "config": config, "collection": name, "dataset_sha256": digest(data),
                "top_k": args.top_k, "threshold": args.threshold,
                "chunks_count": len(chunks), "summary": summary, "rows": rows,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Отчёт: {args.out}")
    finally:
        client.close()


if __name__ == "__main__":
    main()