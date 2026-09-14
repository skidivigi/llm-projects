SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

UV ?= uv
PYTHON_VERSION ?= 3.13.7
QUERY ?= Как подключиться к рабочей сети из дома?
TOP_K ?= 2
THRESHOLD ?= 0.85
SIZE ?= 128
OVERLAP ?= 0
DOC ?= backup_prod
QUESTION ?= q11

# Передаём текст и параметры через окружение, чтобы кавычки в вопросе не ломали shell.
export QUERY TOP_K THRESHOLD SIZE OVERLAP DOC QUESTION PYTHON_VERSION

.PHONY: help setup up status logs ready gpu-check
.PHONY: lab1 lab1-check lab1-query lab1-backup lab1-vacation
.PHONY: lab2 lab2-index lab2-search lab2-search-raw lab2-baseline
.PHONY: lab2-k1 lab2-k2 lab2-t080 lab2-t085 lab2-eval lab2-compare
.PHONY: lab3 lab3-prepare lab3-inspect lab3-index lab3-index-one
.PHONY: lab3-eval lab3-eval-one lab3-question lab3-question-raw lab3-compare

### COMMON

help:
	@printf '%s\n' \
	  'Общее: setup | up | status | logs | ready | gpu-check' \
	  'Урок 1: lab1 | lab1-query QUERY="..." | lab1-backup | lab1-vacation' \
	  'Урок 2: lab2 | lab2-index | lab2-eval | lab2-compare' \
	  '  lab2-search QUERY="..." TOP_K=2 THRESHOLD=0.85' \
	  '  lab2-search-raw QUERY="..." TOP_K=3 (без порога)' \
	  '  lab2-baseline | lab2-k1 | lab2-k2 | lab2-t080 | lab2-t085' \
	  'Урок 3: lab3 | lab3-prepare | lab3-index | lab3-eval | lab3-compare' \
	  '  lab3-inspect SIZE=128 OVERLAP=32 DOC=backup_prod' \
	  '  lab3-index-one SIZE=128 OVERLAP=32' \
	  '  lab3-eval-one SIZE=128 OVERLAP=32 TOP_K=2 THRESHOLD=0.85' \
	  '  lab3-question QUESTION=q11 SIZE=128 OVERLAP=0 TOP_K=2 THRESHOLD=0.85' \
	  '  lab3-question-raw QUESTION=q11 TOP_K=4 (без порога)'

# Требуются установленные pyenv и uv; повторять после каждого урока не нужно.
setup:
	pyenv install -s "$$PYTHON_VERSION"
	pyenv local "$$PYTHON_VERSION"
	$(UV) sync --python "$$(pyenv which python)"

up:
	docker compose up -d

status:
	docker compose ps

# Последние сообщения без постоянно открытого потока.
logs:
	docker compose logs --tail=80 qdrant

ready:
	$(UV) run python -c 'from check_env import wait_for_qdrant; wait_for_qdrant(); print("Qdrant: OK")'

gpu-check:
	$(UV) run python - <<'PY'
	import torch
	print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
	assert torch.cuda.is_available(), "CUDA недоступна"
	print("GPU:", torch.cuda.get_device_name(0))
	x = torch.randn(1024, 1024, device="cuda")
	y = x @ x
	torch.cuda.synchronize()
	print("Вычисление на GPU: OK; устройство:", y.device)
	PY

### 1ST LAB

lab1: lab1-check

lab1-check: up
	$(UV) run python check_env.py

lab1-query:
	$(UV) run python check_env.py "$$QUERY"

lab1-backup:
	$(UV) run python check_env.py "Сколько времени сохраняются бэкапы?"

lab1-vacation:
	$(UV) run python check_env.py "С кем согласовать даты отдыха?"

### 2ND LAB

# Последовательность сохраняется и при запуске make -j.
lab2: lab2-index
	$(MAKE) lab2-eval
	$(MAKE) lab2-compare

lab2-index: up
	$(UV) run python index.py

lab2-search:
	$(UV) run python search.py --query "$$QUERY" --top-k "$$TOP_K" --threshold "$$THRESHOLD"

lab2-search-raw:
	$(UV) run python search.py --query "$$QUERY" --top-k "$$TOP_K"

# Фиксированные настройки опытов из урока 2.
lab2-baseline:
	$(UV) run python search.py --eval --top-k 3 --out reports/baseline_k3.json

lab2-k1:
	$(UV) run python search.py --eval --top-k 1 --out reports/k1.json

lab2-k2:
	$(UV) run python search.py --eval --top-k 2 --threshold 0.85 --out reports/k2_t085.json

lab2-t080:
	$(UV) run python search.py --eval --top-k 3 --threshold 0.80 --out reports/k3_t080.json

lab2-t085:
	$(UV) run python search.py --eval --top-k 3 --threshold 0.85 --out reports/k3_t085.json

lab2-eval:
	$(MAKE) lab2-baseline
	$(MAKE) lab2-k1
	$(MAKE) lab2-t080
	$(MAKE) lab2-t085
	$(MAKE) lab2-k2

define LAB2_COMPARE
import json
from pathlib import Path
names = ["baseline_k3.json", "k1.json", "k3_t080.json", "k3_t085.json", "k2_t085.json"]
print(f"{'Прогон':<22} {'Recall':>8} {'MRR':>8} {'Шум':>8} {'Ложн.выдача':>12}")
for name in names:
    path = Path("reports") / name
    if not path.exists():
        print(f"{name:<22} нет отчёта; запусти make lab2-eval")
        continue
    s = json.loads(path.read_text(encoding="utf-8"))["summary"]
    print(f"{name:<22} {s['recall_at_k']:>8.3f} {s['mrr_at_k']:>8.3f} {s['mean_noise']:>8.3f} {s['false_accept_rate']:>12.3f}")
endef
export LAB2_COMPARE

lab2-compare:
	@$(UV) run python -c "$$LAB2_COMPARE"

### 3RD LAB

lab3: lab3-prepare
	$(MAKE) lab3-index
	$(MAKE) lab3-eval
	$(MAKE) lab3-compare

# Повторно создаёт chunk_dataset.json из исходного набора шага 2.
lab3-prepare:
	$(UV) run python prepare_chunks.py

lab3-inspect:
	$(UV) run python chunk_lab.py inspect --size "$$SIZE" --overlap "$$OVERLAP" --doc "$$DOC"

lab3-index: up
	for spec in 64:0 128:0 256:0 128:32; do
	    size=$${spec%:*}
	    overlap=$${spec#*:}
	    $(UV) run python chunk_lab.py index --size "$$size" --overlap "$$overlap"
	done

lab3-index-one: up
	$(UV) run python chunk_lab.py index --size "$$SIZE" --overlap "$$OVERLAP"

# TOP_K и THRESHOLD одинаковы для всех четырёх вариантов в одном запуске.
lab3-eval:
	for spec in 64:0 128:0 256:0 128:32; do
	    size=$${spec%:*}
	    overlap=$${spec#*:}
	    $(UV) run python chunk_lab.py eval --size "$$size" --overlap "$$overlap" \
	      --top-k "$$TOP_K" --threshold "$$THRESHOLD" \
	      --out "reports/chunks/s$${size}_o$${overlap}.json"
	done

# Отдельный опыт не перезаписывает четыре отчёта основной таблицы.
lab3-eval-one:
	$(UV) run python chunk_lab.py eval --size "$$SIZE" --overlap "$$OVERLAP" \
	  --top-k "$$TOP_K" --threshold "$$THRESHOLD" \
	  --out "reports/chunks/manual_s$${SIZE}_o$${OVERLAP}_k$${TOP_K}_t$${THRESHOLD}.json"

lab3-question:
	$(UV) run python chunk_lab.py eval --size "$$SIZE" --overlap "$$OVERLAP" \
	  --top-k "$$TOP_K" --threshold "$$THRESHOLD" --question "$$QUESTION"

lab3-question-raw:
	$(UV) run python chunk_lab.py eval --size "$$SIZE" --overlap "$$OVERLAP" \
	  --top-k "$$TOP_K" --question "$$QUESTION"

define LAB3_COMPARE
import json
from pathlib import Path
names = ["s64_o0", "s128_o0", "s256_o0", "s128_o32"]
signatures = set()
print(f"{'Вариант':<12} {'k':>3} {'Порог':>6} {'Чанков':>7} {'DocRec':>7} {'EvRec':>7} {'Ложн.':>7} {'Дубли':>7} {'Токены':>8}")
for name in names:
    path = Path("reports/chunks") / f"{name}.json"
    if not path.exists():
        print(f"{name:<12} нет отчёта; запусти make lab3-eval")
        continue
    r = json.loads(path.read_text(encoding="utf-8"))
    s = r["summary"]
    signatures.add((r["dataset_sha256"], r["top_k"], r["threshold"]))
    print(f"{name:<12} {r['top_k']:>3} {str(r['threshold']):>6} {r['chunks_count']:>7} "
          f"{s['doc_recall']:>7.3f} {s['evidence_recall']:>7.3f} {s['false_accept_rate']:>7.3f} "
          f"{s['mean_duplicate_share']:>7.3f} {s['mean_context_tokens']:>8.1f}")
if len(signatures) > 1:
    print("В таблице разные данные, k или пороги. Для сопоставимого прогона выполни make lab3-eval.")
endef
export LAB3_COMPARE

lab3-compare:
	@$(UV) run python -c "$$LAB3_COMPARE"
