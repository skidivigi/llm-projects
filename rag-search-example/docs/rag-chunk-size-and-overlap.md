# Шаг 3. Размер чанков, overlap и полнота найденного ответа

## 1. Что добавляем и зачем

На предыдущем шаге k=2 и threshold=0.85 сохранили все нужные документы и оставили три лишних фрагмента на 14 вопросов. Теперь проверим эти настройки на более длинных документах.

Результат шага: четыре варианта разбиения одного набора, четыре отчёта и понимание того, где поиск теряет фрагмент с ответом или заполняет выдачу повторяющимся текстом.

Стек остаётся прежним: Python 3.13 через pyenv, uv, CUDA PyTorch, multilingual-e5-small и Qdrant. Команды выполняются из корня существующего проекта rag-search-example; далее `./` означает именно этот каталог. Новые зависимости не нужны.

### Главная разница с прошлым шагом

Раньше документ целиком был одним чанком. Теперь найденный кусок нужного документа может содержать только вступление, а нужный срок или инструкция окажется в соседнем чанке. Поэтому проверяем два разных события:

- **DocRecall:** нашли ли хотя бы один чанк каждого нужного документа.
- **EvidenceRecall:** покрывают ли найденные чанки размеченные фрагменты с ответом.

Например: нашли вступление к регламенту prod — DocRecall=1, но если предложение со сроком хранения не попало в выдачу, EvidenceRecall=0.

Эталонный фрагмент считается покрытым, когда его текст целиком присутствует в объединении найденных диапазонов одного документа. Два соседних чанка вместе могут его покрыть. Частично найденный эталонный фрагмент не засчитывается. Это строгая проверка покрытия заданного текста, а не оценка ответа LLM или всех возможных перефразировок.

Для q11 нужны два эталонных фрагмента: сроки prod и dev. Найден только один — EvidenceRecall этого вопроса равен 0.5.

### Что сравниваем

| Вариант | Размер окна исходных токенов | Overlap | Назначение |
|---|---:|---:|---|
| s64_o0 | 64 | 0 | Маленькие фрагменты |
| s128_o0 | 128 | 0 | Средние фрагменты |
| s256_o0 | 256 | 0 | Более широкий контекст |
| s128_o32 | 128 | 32 | Средние фрагменты с перекрытием |

Размер считаем токенизатором embedding-модели. При size=128 и overlap=32 следующее окно начинается через 96 исходных токенов. Окна: [0,128), [96,224), [192,320). Последнее окно может быть короче.

Используем координаты токенов в исходном тексте, чтобы сохранять точные подстроки, включая пробелы между окнами. Fast tokenizer поддерживает такое сопоставление. При повторной токенизации выделенного чанка количество токенов может немного отличаться из-за новых границ слова. В `input_tokens` сохраняем фактическую длину входа, включая префикс и специальные токены. [Документация tokenizer](https://huggingface.co/docs/transformers/main_classes/tokenizer)

Перед embedding проверяем каждый полный вход `passage: <текст>` на лимит модели; слишком длинный вход вызывает ошибку. Модель E5 и её префиксы сохраняются. [Карточка модели](https://huggingface.co/intfloat/multilingual-e5-small)

Большие чанки могут сохранить контекст, маленькие — точнее выделить тему. Overlap может помочь на границе, но также создать повторяющиеся результаты. Улучшение от него заранее не гарантировано.

## 2. Файлы и полный код

Добавь два файла ниже в корень проекта. Или распакуй rag-lab-step-03.zip туда: архив содержит эти же два файла.

Создавать каталог `data` повторно не нужно: он появился в шаге 2. Существующие `data/dataset.json`, `lab_common.py` и `check_env.py` используются как основа. `index.py` и `search.py` шага 2 продолжают работать с прежним набором.

### Создать `./prepare_chunks.py`

Скрипт добавляет к каждому из 15 документов два содержательных абзаца и сохраняет новый набор в `./data/chunk_dataset.json`. Исходный фрагмент располагается в начале, середине или конце документа. Все данные вымышленные. Контрольные вопросы сохраняются, а для каждого добавляются точные диапазоны текста с ответом.

Выполнение скрипта повторно перезаписывает только `data/chunk_dataset.json` из исходного набора и заданных расширений.

```python
import json

from lab_common import ROOT, load_dataset

# Дополнительные абзацы вымышленной компании: область применения и детали процесса.
EXTRA = {
    "vpn_employee": [
        "Удалённое рабочее место используется для доступа к внутренним приложениям вне офиса. До подключения сотрудник проверяет интернет и устанавливает обновления операционной системы. Публичные сайты могут открываться даже при отсутствии доступа к корпоративным ресурсам.",
        "После завершения работы соединение следует отключить. Если отдельное приложение не открывается, в обращении указывают его название и время сбоя. Наличие VPN не выдаёт автоматически права на каждую информационную систему: их согласует владелец ресурса."
    ],
    "vpn_contractor": [
        "Подрядчик выполняет работы в рамках конкретного проекта. До начала подключения куратор уточняет перечень систем, период работ и контакт ответственного со стороны исполнителя. Эти сведения нужны для ограничения доступа только необходимыми ресурсами.",
        "При завершении договора куратор направляет заявку на прекращение доступа. Продление работ требует повторного согласования срока. Наличие доступа у другого участника проекта не означает, что новому специалисту разрешены те же системы."
    ],
    "vpn_error": [
        "При диагностике подключения сначала зафиксируйте точный код ошибки и время её появления. Сообщения о неверном пароле и об отсутствии сети требуют разных действий. Не удаляйте настройки клиента до выяснения причины отказа.",
        "После изменения учётных данных удалите старое сохранённое значение в клиенте. Если подключение снова не удалось, приложите снимок сообщения без секретных данных. Номер версии клиента помогает специалисту воспроизвести проблему на аналогичной конфигурации."
    ],
    "otp_reset": [
        "Приложение подтверждения входа привязывается к конкретному устройству пользователя. Копирование ярлыка приложения на другой телефон не переносит регистрацию фактора. При потере устройства важно сообщить об этом, даже если основной пароль сохранился.",
        "После регистрации нового фактора выполните пробный вход в корпоративную систему. Не отправляйте коды и изображения секретов регистрации другим людям. Если старый телефон впоследствии найден, его прежняя регистрация не должна автоматически восстанавливаться."
    ],
    "password_portal": [
        "Кадровый портал используется для личных заявлений и просмотра статуса согласований. На странице входа проверьте раскладку клавиатуры и правильность имени пользователя. Не создавайте вторую учётную запись для обхода проблемы со входом.",
        "После восстановления доступа убедитесь, что открывается ваш личный профиль. Повторная отправка формы может сделать прежнюю ссылку неактуальной. Если письмо не найдено, проверьте папку нежелательной почты и правильность выбранной учётной записи."
    ],
    "password_db": [
        "Сервисные учётные записи используются приложениями для автоматического подключения к данным. Владелец приложения заранее согласует окно изменения секрета с администратором. Перед работами составляют перечень всех сервисов, использующих эти реквизиты.",
        "После изменения проверяют подключения и журнал ошибок приложения. Если один экземпляр сервиса продолжает использовать старые реквизиты, его конфигурацию обновляют отдельно. Секрет не следует сохранять в открытом виде в заявке или журнале диагностики."
    ],
    "backup_prod": [
        "Производственная база обслуживает рабочие операции компании. Регламент резервирования распространяется на основную базу и проверяется владельцем системы. Успешное завершение задания подтверждается журналом; наличие файла без проверки журнала не считается подтверждением успешной копии.",
        "Контроль восстановления проводится в отдельном окружении. Результаты проверки заносят в журнал эксплуатации, включая имя базы и выбранную точку восстановления. Перед использованием копии необходимо убедиться, что она относится к нужной системе, а не к похожему тестовому экземпляру."
    ],
    "backup_dev": [
        "Тестовая база используется для разработки и проверки изменений. Набор данных в ней может отличаться от производственного. Успешность резервирования проверяют по журналу задания, отдельно от отчёта об успешном развёртывании приложения в тестовом окружении.",
        "Восстановление тестовых данных согласуют с командой разработки, чтобы не прервать текущие проверки. Перед операцией участникам сообщают о возможной потере последних изменений. Для испытания новой версии приложения можно выбрать отдельный тестовый экземпляр базы."
    ],
    "restore_db": [
        "Восстановление меняет состояние данных и требует предварительной подготовки. Инициатор определяет причину операции и проверяет, какие приложения используют выбранную базу. Название системы само по себе не заменяет точное имя экземпляра базы.",
        "Администратор проверяет доступность выбранной копии и сообщает план работ. После операции владелец проверяет целостность прикладных данных и доступность основных функций. Завершение технической операции ещё не означает, что бизнес-проверка пройдена."
    ],
    "vacation": [
        "Планирование отсутствия помогает команде распределить задачи и назначить замещение. Перед отправкой заявления сотрудник проверяет календарь проекта и сообщает коллегам о предполагаемом периоде. Устное обсуждение не заменяет регистрацию заявления в системе.",
        "При изменении планов обновите заявление и дождитесь нового согласования. До ухода передайте текущие задачи замещающему сотруднику. Статус заявки можно проверить в личном кабинете; уведомление о создании заявки ещё не подтверждает её одобрение."
    ],
    "business_trip": [
        "Рабочая поездка должна быть связана с задачами подразделения. Перед оформлением сотрудник уточняет место проведения встречи и доступные способы проезда. Бронирование билетов проводится после проверки планов, чтобы избежать расходов при переносе мероприятия.",
        "После возвращения сотрудник передаёт подтверждающие документы в бухгалтерию. При изменении маршрута или продолжительности поездки сведения в заявке обновляются. Личная поездка во время отдыха не оформляется как служебная командировка."
    ],
    "laptop": [
        "Рабочее устройство закрепляется за сотрудником по инвентарному номеру. При неисправности важно описать наблюдаемые признаки: отсутствие индикации, ошибки загрузки или внезапное выключение. Самостоятельная разборка устройства затрудняет последующую диагностику.",
        "При передаче оборудования специалист фиксирует комплектность и внешнее состояние. Пользователь сообщает о незавершённых задачах и расположении рабочих данных. Возврат отремонтированного устройства оформляется после проверки включения и основных рабочих приложений."
    ],
    "monitor": [
        "Дополнительное оборудование выдаётся для организации рабочего места. При планировании учитывают доступные разъёмы ноутбука, место на столе и необходимость переходника. Несовместимый кабель может помешать подключению даже исправного устройства.",
        "При получении сотрудник проверяет изображение и комплектность. Если устройство выдаётся для временного рабочего места, срок использования указывают в заявке. Передача монитора другому подразделению оформляется через учёт оборудования."
    ],
    "guest_wifi": [
        "Гостевое подключение предназначено для посетителей переговорных и участников внешних встреч. Принимающий сотрудник сообщает гостю правила использования сети. Подключение личного устройства посетителя не требует создания штатной корпоративной учётной записи.",
        "Если страница авторизации не появилась, посетитель может повторно открыть браузер после подключения. Наличие интернета не означает доступность внутренних приложений. Для рабочих ресурсов внешнего исполнителя требуется отдельное согласование с куратором проекта."
    ],
    "employee_wifi": [
        "Корпоративная беспроводная сеть используется на рабочих местах и в переговорных. Сотрудник проверяет выбранное имя сети перед вводом реквизитов. Сохранённое подключение к гостевой сети может мешать автоматическому выбору нужного профиля.",
        "При смене рабочего пароля обновите сохранённые реквизиты подключения. Если проблема возникает только в одной зоне офиса, укажите помещение в обращении. Доступность сети и наличие прав на отдельное приложение проверяются независимо."
    ]
}

EVIDENCE = {
    "q01": [("vpn_employee", "Сотрудник подключается к рабочей сети из дома через клиент CorpVPN. Укажите vpn.company.test, введите рабочий логин и пароль, затем одноразовый код из приложения.")],
    "q02": [("vpn_contractor", "Внешнему подрядчику удалённый доступ оформляет куратор проекта через заявку в Service Desk. После согласования выдаётся отдельная учётная запись с ограничением по сроку и доступным системам.")],
    "q03": [("otp_reset", "Если телефон с приложением одноразовых кодов потерян, обратитесь в Service Desk для сброса MFA. После проверки личности специалист отключит старый фактор и поможет привязать новое устройство.")],
    "q04": [("password_portal", "Если вы забыли пароль кадрового портала, выберите Восстановить пароль на странице входа. Ссылка для сброса поступит на рабочую почту.")],
    "q05": [("vpn_error", "Ошибка E104 при подключении CorpVPN означает, что срок действия пароля истёк. Смените пароль через портал самообслуживания и повторите подключение к VPN. Переустановка VPN-клиента для этой ошибки не требуется.")],
    "q06": [("backup_prod", "Копии хранятся 30 дней.")],
    "q07": [("backup_dev", "Копии хранятся 7 дней.")],
    "q08": [("restore_db", "Укажите имя базы, окружение prod или dev и требуемую дату восстановления. Для prod необходимо согласование владельца системы.")],
    "q09": [("vacation", "Даты согласует непосредственный руководитель.")],
    "q10": [("laptop", "На время ремонта можно запросить подменный ноутбук.")],
    "q11": [("backup_prod", "Копии хранятся 30 дней."), ("backup_dev", "Копии хранятся 7 дней.")],
    "q12": [("guest_wifi", "Временный код доступа выдаёт сотрудник ресепшена.")],
    "q13": [],
    "q14": []
}


def main():
    data = load_dataset()
    if {doc["id"] for doc in data["documents"]} != set(EXTRA):
        raise SystemExit("Ожидаются 15 документов из шага 2; проверь data/dataset.json")
    if {q["id"] for q in data["questions"]} != set(EVIDENCE):
        raise SystemExit("Ожидаются 14 вопросов из шага 2")
    for i, doc in enumerate(data["documents"]):
        paragraphs = EXTRA[doc["id"]].copy()
        paragraphs.insert(i % 3, doc["text"])
        doc["text"] = "\n\n".join(paragraphs)
    docs = {doc["id"]: doc for doc in data["documents"]}
    for question in data["questions"]:
        question["evidence"] = []
        for doc_id, quote in EVIDENCE[question["id"]]:
            text = docs[doc_id]["text"]
            if text.count(quote) != 1:
                raise SystemExit(f"{question['id']}: эталон должен встретиться ровно один раз")
            start = text.index(quote)
            question["evidence"].append({
                "doc_id": doc_id, "quote": quote, "start": start, "end": start + len(quote)
            })
    path = ROOT / "data/chunk_dataset.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Создан {path.name}: 15 документов, 14 вопросов, размеченные фрагменты ответа")


if __name__ == "__main__":
    main()
```

### Создать `./chunk_lab.py`

Один CLI с тремя отдельными действиями:

- `inspect` показывает границы и текст чанков, используя только токенизатор.
- `index` рассчитывает embeddings чанков и записывает их в Qdrant.
- `eval` рассчитывает embeddings вопросов и читает коллекцию, затем сохраняет отчёт.

Заголовки хранятся как метаданные: к embedding-тексту их в этом опыте не добавляем. `k` ограничивает количество чанков; объединения или ограничения по документам пока нет. Точный поиск и cosine сохраняются.

Имя коллекции зависит от содержания чанков и параметров разбиения. Каждому варианту соответствует своя коллекция. Изменение только k или threshold переиндексации не требует.

```python
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
```

## 3. Структура проекта

| Относительный путь | Назначение |
|---|---|
| `./prepare_chunks.py` | Новый: подготовка длинных документов и эталонных диапазонов |
| `./chunk_lab.py` | Новый: разбиение, индексация, оценка |
| `./data/chunk_dataset.json` | Генерируется: 15 расширенных документов и 14 вопросов |
| `./reports/chunks/` | Создаётся при сохранении результатов |
| `./data/dataset.json` | Исходный набор шага 2 |
| `./lab_common.py`, `./check_env.py` | Существующие настройки, модель и подключение |

## 4. Запуск

Все команды — из корня проекта, в прежнем терминале WSL с настроенными pyenv и uv.

### Подготовить документы

```bash
docker compose up -d
uv run python prepare_chunks.py
```

Ожидаем:

```text
Создан chunk_dataset.json: 15 документов, 14 вопросов, размеченные фрагменты ответа
```

Если скрипт сообщает, что эталон не найден или список ID отличается, проверь, не изменял ли ты исходные документы шага 2. Разметка предполагает их исходные формулировки. Для расширений редактируй EXTRA, для эталонов — EVIDENCE в prepare_chunks.py, затем генерируй набор заново.

### Посмотреть, что реально попадает в чанк

```bash
uv run python chunk_lab.py inspect --size 64 --overlap 0 --doc backup_prod
uv run python chunk_lab.py inspect --size 128 --overlap 0 --doc backup_prod
uv run python chunk_lab.py inspect --size 128 --overlap 32 --doc backup_prod
```

У каждого чанка будут ID, границы исходных токенов и символов, фактическая длина входа и текст. Посмотри, в каком чанке находится срок хранения и повторяется ли он в соседнем. Если именно у backup_prod граница не задела важное предложение, посмотри `--doc vpn_error` или `--doc vpn_contractor`.

### Индексировать четыре варианта

```bash
uv run python chunk_lab.py index --size 64 --overlap 0
uv run python chunk_lab.py index --size 128 --overlap 0
uv run python chunk_lab.py index --size 256 --overlap 0
uv run python chunk_lab.py index --size 128 --overlap 32
```

Для каждого варианта скрипт выведет отдельное имя коллекции и число записанных точек. Число точек определяется токенизацией и обычно отличается между вариантами. На неизменных данных повторная индексация обновляет те же точки без дублей.

### Сравнить разбиение при k=2 и threshold=0.85

```bash
uv run python chunk_lab.py eval --size 64 --overlap 0 --top-k 2 --threshold 0.85 --out reports/chunks/s64_o0.json
uv run python chunk_lab.py eval --size 128 --overlap 0 --top-k 2 --threshold 0.85 --out reports/chunks/s128_o0.json
uv run python chunk_lab.py eval --size 256 --overlap 0 --top-k 2 --threshold 0.85 --out reports/chunks/s256_o0.json
uv run python chunk_lab.py eval --size 128 --overlap 32 --top-k 2 --threshold 0.85 --out reports/chunks/s128_o32.json
```

У этих четырёх прогонов один расширенный набор и одинаковые параметры поиска. Сравнение размеров: s64_o0, s128_o0, s256_o0. Сравнение overlap: только s128_o0 и s128_o32, чтобы размер оставался одинаковым.

Не сравнивай эти результаты напрямую с прежним Recall=1 как чистый эффект chunk size: тексты документов также стали длиннее. Исходная точка этого опыта — прогоны на новом наборе.

### Вывести сравнительную таблицу

```bash
uv run python - <<'PYCODE'
import json
from pathlib import Path

names = ['s64_o0', 's128_o0', 's256_o0', 's128_o32']
print(f"{'Вариант':<12} {'Чанков':>7} {'DocRec':>7} {'EvRec':>7} {'Ложн.':>7} {'Дубли':>7} {'Токены':>8}")
for name in names:
    report = json.loads((Path('reports/chunks') / f'{name}.json').read_text(encoding='utf-8'))
    s = report['summary']
    print(
        f"{name:<12} {report['chunks_count']:>7} "
        f"{s['doc_recall']:>7.3f} {s['evidence_recall']:>7.3f} "
        f"{s['false_accept_rate']:>7.3f} {s['mean_duplicate_share']:>7.3f} "
        f"{s['mean_context_tokens']:>8.1f}"
    )
PYCODE
```

Таблица появится после твоего запуска. Ожидаемые численные значения качества заранее не задаём.

## 5. Как прочитать результат и найти причину

| Поле | Что проверяет |
|---|---|
| `doc_recall` | Средняя доля нужных документов, от которых вернулся хотя бы один чанк; вопросы без ответа исключены |
| `evidence_recall` | Средняя доля полностью покрытых эталонных фрагментов ответа; вопросы без ответа исключены |
| `false_accept_rate` | Доля вопросов без ответа, получивших непустую выдачу |
| `mean_duplicate_share` | Средняя доля повторно полученных символов исходного текста в выдаче одного вопроса; учитывается перекрытие диапазонов внутри одного документа |
| `mean_context_tokens` | Средняя сумма токенов найденных чанков, без префиксов и специальных токенов |

Средние для дублей и токенов считаются по всем 14 вопросам; пустая выдача даёт ноль. Дубли здесь — именно повторные диапазоны исходного текста, а не поиск смысловых повторов между разными документами. Сумма токенов чанков — оценка объёма извлечённого текста, не точный размер будущего prompt LLM с заголовками и инструкциями.

**DocRecall=1, EvidenceRecall<1:** нашли правильный источник, но не весь размеченный ответ. Это основная новая ошибка, которую учимся видеть.

**Overlap увеличил число точек и дубли, а EvidenceRecall не вырос:** в этом опыте дополнительное хранение и повторяющийся контекст не окупились улучшением покрытия ответа.

**Большие чанки дали более полный ответ:** проверь также объём контекста. При фиксированном k два больших чанка содержат больше текста, чем два маленьких; это не сравнение при равном бюджете токенов.

**Результат ухудшился после разбиения:** старый порог 0.85 мог стать слишком строгим. Размер чанка меняет вход модели, поэтому могут измениться и score. Для диагностики повтори конкретный вопрос с тем же разбиением, сначала с порогом, затем без него:

```bash
uv run python chunk_lab.py eval --size 128 --overlap 0 --top-k 2 --threshold 0.85 --question q11
uv run python chunk_lab.py eval --size 128 --overlap 0 --top-k 2 --question q11
```

С `--question` скрипт показывает полный текст выдачи и покрытие каждого эталона. Вместо q11 возьми вопрос с потерей ответа из своего отчёта.

- Ответ появился без порога — порог отсеивал нужный чанк.
- Ответа всё ещё нет в двух результатах — это ещё не доказывает, что индекс потерял текст. Проверь ограничение k:

```bash
uv run python chunk_lab.py eval --size 128 --overlap 0 --top-k 4 --question q11
```

Это отдельная диагностическая проверка: её не смешиваем с четырьмя основными прогонами. Без threshold порог не применяется; в основных прогонах он указан явно.

Если два лучших чанка пришли из одного документа, они заняли оба места. В этом шаге мы наблюдаем эффект; ограничение по документам и удаление дублей ещё не добавляем.

При ошибке «Сначала выполни index» используй те же size и overlap, что в eval. При ошибке лимита входа уменьши size; скрипт не обрезает длинный embedding-вход молча. Если Qdrant недоступен:

```bash
docker compose ps
docker compose logs --tail=50 qdrant
```

Проверено при подготовке: синтаксис, соответствие всех эталонов исходному тексту, сохранение хвостов документов и покрытия границ на тестовом токенизаторе, покрытие ответа несколькими соседними чанками, корректность расчёта дублей и изоляция диапазонов разных документов. Реальная токенизация E5, индексация и CUDA-поиск здесь не запускались. Время запросов сохраняется для диагностики; один прогон не является полноценным измерением производительности.

**Критерий завершения:** четыре варианта проиндексированы, четыре отчёта сохранены, получена сравнительная таблица. Пришли таблицу и, если DocRecall отличается от EvidenceRecall, выдачу одного такого вопроса. По ним определим, мешают ли границы чанков, порог или ограничение k.
