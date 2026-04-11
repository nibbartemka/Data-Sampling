# LLM Dataset Service

Объединённый проект на FastAPI, который последовательно выполняет два этапа:

1. Строит representative sample из входного Excel по логике первого разработчика.
2. Передаёт sample и исходный набор в LLM pipeline второго разработчика для построения схемы и выходного датасета.

## Запуск

```bash
uv sync
uvicorn app.main:app --reload
```

Swagger будет доступен по `/docs`.

## Главный endpoint

`POST /api/v1/pipeline/process`

Принимает Excel-файл и возвращает JSON с:
- параметрами запуска;
- краткой статистикой sample;
- схемой выходного датасета;
- строками результата;
- ссылками на сохранённые артефакты (`sample.xlsx`, `schema.json`, `output.json`, `output.xlsx`).

## Требуемые переменные окружения

См. `.env.example`.
