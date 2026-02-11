# Browser Brain: Semantic Search по истории браузера

MVP-проект для семантического поиска по локальной истории браузера (Chrome/Firefox) с использованием эмбеддингов и навигационного графа NSW.

## Возможности
- Импорт истории из SQLite баз Chrome и Firefox.
- Препроцессинг текста и chunking.
- Генерация эмбеддингов (`sentence-transformers`, по умолчанию `all-MiniLM-L6-v2`).
- Построение NSW-графа (short + long edges).
- Семантический поиск через greedy traversal + rerank по cosine similarity.
- CLI для индексации и поиска.

## Установка
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Быстрый старт
### 1) Индексация истории
```bash
python -m browser_brain.cli index \
  --browser chrome \
  --limit 5000 \
  --output-dir .index \
  --chunk-size 260
```

Для Firefox:
```bash
python -m browser_brain.cli index --browser firefox --output-dir .index
```

Можно передать явный путь к sqlite базе:
```bash
python -m browser_brain.cli index --browser chrome --history-path ~/.config/google-chrome/Default/History
```

### 2) Поиск
```bash
python -m browser_brain.cli search \
  --index-dir .index \
  --query "где я смотрел про greedy search в графах?" \
  --top-k 5
```

## Структура индекса
В `output-dir` сохраняются:
- `embeddings.npy` — матрица эмбеддингов (`n_chunks x dim`)
- `chunks.jsonl` — метаданные чанков
- `graph.json` — список соседей NSW
- `meta.json` — параметры индекса

## Примечания
- Если `sentence-transformers` недоступен в окружении, используется детерминированный fallback-эмбеддер на hashing trick (для демонстрации пайплайна).
- Для больших объёмов данных рекомендуется перейти на FAISS HNSW.
