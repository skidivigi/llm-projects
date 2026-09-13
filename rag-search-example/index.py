from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models

from lab_common import collection_name, connect, load_dataset, load_model


def main():
    docs = load_dataset()["documents"]
    name = collection_name(docs)
    model = load_model()
    texts = [f"passage: {doc['text']}" for doc in docs]
    for doc, text in zip(docs, texts):
        length = len(model.tokenizer.encode(text, truncation=False))
        if length > model.max_seq_length:
            raise SystemExit(f"{doc['id']}: {length} токенов; нужен меньший чанк")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    client = connect()
    try:
        if not client.collection_exists(name):
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=int(vectors.shape[1]), distance=models.Distance.COSINE,
                ),
            )
        client.upsert(
            collection_name=name,
            points=[
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, doc["id"])),
                    vector=vector.tolist(), payload=doc,
                )
                for doc, vector in zip(docs, vectors)
            ],
            wait=True,
        )
        print(f"Коллекция: {name}")
        print(f"Точек: {client.count(name, exact=True).count}; размерность: {vectors.shape[1]}")
    finally:
        client.close()


if __name__ == "__main__":
    main()