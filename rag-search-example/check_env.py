import argparse
import time
from urllib.error import URLError
from urllib.request import urlopen

import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

QDRANT_URL = "http://localhost:6333"
MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION = "rag_lab_smoke_e5_small"

# Вымышленные учебные фрагменты, пока каждый целиком является одним чанком.
DOCUMENTS = [
    ("vpn", "Для удалённого доступа к корпоративной сети включите VPN. "
     "Введите рабочий логин и подтвердите вход одноразовым кодом."),
    ("backup", "Резервное копирование базы данных выполняется каждую ночь. "
     "Резервные копии хранятся 30 дней."),
    ("vacation", "Заявку на отпуск подайте через кадровый портал. "
     "Даты отпуска необходимо согласовать с руководителем."),
]


def wait_for_qdrant():
    for _ in range(30):
        try:
            with urlopen(f"{QDRANT_URL}/readyz", timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    raise SystemExit("Qdrant не готов. Проверь: docker compose logs --tail=50 qdrant")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="Как подключиться к рабочей сети из дома?")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA недоступна. Проверь nvidia-smi и CUDA-сборку PyTorch.")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    wait_for_qdrant()
    print("Qdrant: OK", flush=True)
    print(f"Загрузка модели: {MODEL_NAME} (cuda:0)", flush=True)
    model = SentenceTransformer(MODEL_NAME, device="cuda:0")
    print(f"Устройство модели: {model.device}")

    vectors = model.encode(
        [f"passage: {text}" for _, text in DOCUMENTS],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    dimension = int(vectors.shape[1])
    print(f"Векторы документов: {vectors.shape}")

    client = QdrantClient(url=QDRANT_URL, timeout=10)
    try:
        if not client.collection_exists(COLLECTION):
            client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE,
                ),
            )
        config = client.get_collection(COLLECTION).config.params.vectors
        if not isinstance(config, models.VectorParams) or (
            config.size != dimension or config.distance != models.Distance.COSINE
        ):
            raise SystemExit("Конфигурация коллекции отличается. Задай новое имя COLLECTION.")

        client.upsert(
            collection_name=COLLECTION,
            points=[
                models.PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={"doc_id": doc_id, "text": text, "model": MODEL_NAME},
                )
                for i, ((doc_id, text), vector) in enumerate(zip(DOCUMENTS, vectors), start=1)
            ],
            wait=True,
        )
        count = client.count(COLLECTION, exact=True).count
        print(f"Коллекция: {COLLECTION}; точек: {count}")

        torch.cuda.synchronize()
        started = time.perf_counter()
        query_vector = model.encode(f"query: {args.query}", normalize_embeddings=True)
        torch.cuda.synchronize()
        embedding_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        hits = client.query_points(
            collection_name=COLLECTION,
            query=query_vector.tolist(),
            limit=3,
            with_payload=True,
            search_params=models.SearchParams(exact=True),
        ).points
        search_ms = (time.perf_counter() - started) * 1000

        print(f"\nВопрос: {args.query}")
        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}
            print(f"{rank}. {payload.get('doc_id')} | score={hit.score:.4f}")
            print(f"   {payload.get('text')}")
        print(f"\nEmbedding запроса: {embedding_ms:.1f} мс")
        print(f"Запрос к Qdrant: {search_ms:.1f} мс")
        print("Проверка завершена: модель, запись и поиск работают.")
    finally:
        client.close()
        


if __name__ == "__main__":
    main()