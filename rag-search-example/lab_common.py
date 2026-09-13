import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "intfloat/multilingual-e5-small"
QDRANT_URL = "http://localhost:6333"


def load_dataset():
    data = json.loads((ROOT / "data/dataset.json").read_text(encoding="utf-8"))
    docs, questions = data["documents"], data["questions"]
    ids = [doc["id"] for doc in docs]
    if not docs or len(ids) != len(set(ids)):
        raise ValueError("Нужны документы с уникальными id")
    if any(not doc["text"].strip() for doc in docs):
        raise ValueError("Текст документа не может быть пустым")
    if not questions or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Нужны вопросы с уникальными id")
    for question in questions:
        if not set(question["expected"]) <= set(ids):
            raise ValueError(f"Неизвестный expected в {question['id']}")
    return data


def collection_name(docs):
    signature = {"documents": docs, "model": MODEL_NAME, "distance": "Cosine"}
    raw = json.dumps(signature, ensure_ascii=False, sort_keys=True).encode()
    return "rag_lab_step02_" + hashlib.sha256(raw).hexdigest()[:12]


def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна. Проверь nvidia-smi и PyTorch.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    return SentenceTransformer(MODEL_NAME, device="cuda:0")


def connect():
    from qdrant_client import QdrantClient
    from check_env import wait_for_qdrant

    wait_for_qdrant()
    return QdrantClient(url=QDRANT_URL, timeout=10)


def score_result(expected, found):
    relevant = set(expected)
    ranks = [i for i, doc_id in enumerate(found, 1) if doc_id in relevant]
    return {
        "recall": len(relevant & set(found)) / len(relevant) if relevant else None,
        "rr": 1 / ranks[0] if ranks else (0.0 if relevant else None),
        "noise": sum(doc_id not in relevant for doc_id in found),
    }