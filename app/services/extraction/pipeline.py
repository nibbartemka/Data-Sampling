from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable

import pandas as pd

from app.services.extraction.yandex_client import YandexGPTClient


logger = logging.getLogger(__name__)


class FeatureExtractionPipeline:
    ALLOWED_COLUMN_TYPES = {'id', 'binary', 'numeric', 'categorical'}
    NULL_STRINGS = {'', 'null', 'none', 'nan', 'n/a', 'na', 'not_provided', 'unknown'}
    INTERNAL_ROW_UID = '__row_uid'

    def __init__(self, yandex_client: YandexGPTClient, schema_prompt: str, batch_prompt: str) -> None:
        self._yandex_client = yandex_client
        self._schema_prompt = schema_prompt
        self._batch_prompt = batch_prompt

    def run(
        self,
        sample_records: list[dict[str, Any]],
        raw_records: list[dict[str, Any]],
        batch_size: int = 10,
        schema_max_tokens: int = 6000,
        batch_max_tokens: int = 6000,
        temperature: float = 0.1,
        domain_description: str | None = None,
    ) -> dict[str, Any]:
        if not sample_records:
            raise ValueError('Для построения схемы нужен непустой sample_records')
        if not raw_records:
            raise ValueError('Для извлечения признаков нужен непустой raw_records')

        logger.info('Запуск extraction pipeline. sample=%s raw=%s', len(sample_records), len(raw_records))
        schema_json = self.build_schema(sample_records, schema_max_tokens, temperature, domain_description)
        prepared_raw_records = self._attach_row_uids(raw_records)
        all_rows: list[dict[str, Any]] = []
        batches = list(self.split_batches(prepared_raw_records, batch_size))
        for index, batch in enumerate(batches, start=1):
            logger.info('Обработка batch %s/%s size=%s', index, len(batches), len(batch))
            batch_rows = self.normalize_batch(schema_json, batch, batch_max_tokens, temperature, domain_description)
            all_rows.extend(batch_rows)
        return {'schema': schema_json, 'rows': all_rows}

    def build_schema(
        self,
        sample_records: list[dict[str, Any]],
        max_tokens: int = 6000,
        temperature: float = 0.1,
        domain_description: str | None = None,
    ) -> dict[str, Any]:
        domain_block = (
            f'domain_description: {domain_description}\n'
            if domain_description
            else 'domain_description: NOT_PROVIDED\n'
        )
        user_text = (
            'Входные данные:\n'
            f'{domain_block}'
            f'sample: {json.dumps(sample_records, ensure_ascii=False)}\n'
            'raw: RAW_IS_NOT_TO_BE_PROCESSED_ON_THIS_STAGE'
        )
        messages = [{'role': 'system', 'text': self._schema_prompt}, {'role': 'user', 'text': user_text}]
        response_text = self._yandex_client.call(messages=messages, max_tokens=max_tokens, temperature=temperature)
        schema_json = self.safe_json_loads(response_text)
        if not isinstance(schema_json, dict):
            raise ValueError('Ожидался JSON-объект схемы, но модель вернула другой формат')
        schema_json = self._normalize_schema(schema_json)
        self._validate_schema(schema_json)
        return schema_json

    def normalize_batch(
        self,
        schema_json: dict[str, Any],
        batch: list[dict[str, Any]],
        max_tokens: int = 6000,
        temperature: float = 0.1,
        domain_description: str | None = None,
    ) -> list[dict[str, Any]]:
        domain_block = (
            f'domain_description: {domain_description}\n'
            if domain_description
            else 'domain_description: NOT_PROVIDED\n'
        )
        user_text = (
            'Входные данные:\n'
            f'{domain_block}'
            f'schema_json: {json.dumps(schema_json, ensure_ascii=False)}\n'
            f'raw_batch: {json.dumps(batch, ensure_ascii=False)}'
        )
        messages = [{'role': 'system', 'text': self._batch_prompt}, {'role': 'user', 'text': user_text}]
        response_text = self._yandex_client.call(messages=messages, max_tokens=max_tokens, temperature=temperature)
        batch_rows = self.safe_json_loads(response_text)
        if not isinstance(batch_rows, list):
            raise ValueError('Ожидался JSON-массив объектов для batch, но пришло что-то другое')
        return self._normalize_batch_rows(schema_json, batch, batch_rows)

    @staticmethod
    def clean_json_text(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[len('```json'):].strip()
        elif cleaned.startswith('```'):
            cleaned = cleaned[len('```'):].strip()
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3].strip()
        return cleaned

    @classmethod
    def safe_json_loads(cls, text: str) -> Any:
        cleaned = cls.clean_json_text(text)
        cleaned = re.sub(r'\bNULL\b', 'null', cleaned)
        cleaned = re.sub(r'\bTrue\b', 'true', cleaned)
        cleaned = re.sub(r'\bFalse\b', 'false', cleaned)
        return json.loads(cleaned)

    @staticmethod
    def split_batches(records: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
        if batch_size <= 0:
            raise ValueError('batch_size должен быть > 0')
        for idx in range(0, len(records), batch_size):
            yield records[idx: idx + batch_size]

    @classmethod
    def make_json_serializable(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: cls.make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls.make_json_serializable(v) for v in obj]
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(obj, 'item'):
            try:
                obj = obj.item()
            except Exception:  # noqa: BLE001
                pass
        try:
            if pd.isna(obj):
                return None
        except Exception:  # noqa: BLE001
            pass
        return obj

    @classmethod
    def _attach_row_uids(cls, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            item = dict(record)
            item[cls.INTERNAL_ROW_UID] = str(index)
            prepared.append(item)
        return prepared

    @staticmethod
    def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    @classmethod
    def apply_schema_to_dataframe(cls, rows: list[dict[str, Any]], schema: dict[str, Any] | None = None) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if schema:
            final_columns = list(schema.get('final_columns', []))
            display_names = schema.get('display_names', {})
            for column in final_columns:
                if column not in df.columns:
                    df[column] = None
            if final_columns:
                other_columns = [col for col in df.columns if col not in final_columns]
                df = df[final_columns + other_columns]
            rename_map = {col: display_names[col] for col in df.columns if col in display_names}
            df = df.rename(columns=rename_map)
        return df

    @classmethod
    def _normalize_batch_rows(
        cls,
        schema_json: dict[str, Any],
        batch: list[dict[str, Any]],
        batch_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        id_field = schema_json['id_field']
        final_columns = list(schema_json['final_columns'])
        column_types = schema_json.get('column_types', {})

        if len(batch_rows) != len(batch):
            raise ValueError(
                f'LLM вернула {len(batch_rows)} строк для batch размера {len(batch)}. Требуется 1:1 соответствие.'
            )
        if not all(isinstance(row, dict) for row in batch_rows):
            raise ValueError('Каждый элемент batch-ответа должен быть JSON-объектом')

        source_row_uids = [str(record.get(cls.INTERNAL_ROW_UID)) for record in batch]
        response_row_uids = [
            None if row.get(cls.INTERNAL_ROW_UID) in (None, '', 'NULL', 'null') else str(row.get(cls.INTERNAL_ROW_UID))
            for row in batch_rows
        ]
        ordered_rows = batch_rows
        present_row_uids = [row_uid for row_uid in response_row_uids if row_uid is not None]

        if len(present_row_uids) == len(batch_rows):
            if len(set(present_row_uids)) != len(present_row_uids):
                raise ValueError('LLM вернула duplicate __row_uid в batch-ответе')
            if set(present_row_uids) != set(source_row_uids):
                raise ValueError('LLM вернула __row_uid, которые не соответствуют входному batch')
            index_by_row_uid = {str(row[cls.INTERNAL_ROW_UID]): row for row in batch_rows}
            ordered_rows = [index_by_row_uid[str(record.get(cls.INTERNAL_ROW_UID))] for record in batch]
        elif present_row_uids:
            raise ValueError('LLM вернула __row_uid только для части строк batch — такой ответ нельзя безопасно сопоставить')
        else:
            expected_ids = [str(record.get(id_field)) for record in batch]
            if len(set(expected_ids)) != len(expected_ids):
                raise ValueError(
                    'Во входном batch есть повторяющиеся id. Для безопасного сопоставления LLM обязана вернуть __row_uid для каждой строки'
                )
            normalized_response_ids = [
                None if row.get(id_field) in (None, '', 'NULL', 'null') else str(row.get(id_field))
                for row in batch_rows
            ]
            present_ids = [rid for rid in normalized_response_ids if rid is not None]
            if len(present_ids) == len(batch_rows):
                if len(set(present_ids)) == len(present_ids) and set(present_ids) == set(expected_ids):
                    index_by_id = {str(row[id_field]): row for row in batch_rows}
                    ordered_rows = [index_by_id[str(record.get(id_field))] for record in batch]
                elif present_ids != expected_ids:
                    raise ValueError(
                        'LLM вернула batch-ответ с несовпадающими id и его нельзя безопасно сопоставить с входными строками'
                    )
            elif present_ids:
                raise ValueError('LLM вернула id только для части строк batch — такой ответ нельзя безопасно сопоставить')

        normalized_rows: list[dict[str, Any]] = []
        for source_record, raw_row in zip(batch, ordered_rows, strict=True):
            clean_row: dict[str, Any] = {}
            source_id = cls.make_json_serializable(source_record.get(id_field))
            returned_id = raw_row.get(id_field)
            clean_row[id_field] = source_id if returned_id in (None, '', 'NULL', 'null') else cls.make_json_serializable(returned_id)
            if str(clean_row[id_field]) != str(source_id):
                logger.warning('LLM вернула несовпадающий id. Использую исходный id=%s вместо %s', source_id, returned_id)
                clean_row[id_field] = source_id

            for column in final_columns:
                if column == id_field:
                    continue
                clean_row[column] = cls._coerce_value(raw_row.get(column), column_types.get(column))
            normalized_rows.append(clean_row)
        return normalized_rows


    @classmethod
    def _normalize_schema(cls, schema_json: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(schema_json)

        id_field = normalized.get('id_field')
        if not isinstance(id_field, str) or not id_field.strip():
            raise ValueError('schema.id_field должен быть непустой строкой')
        id_field = id_field.strip()
        normalized['id_field'] = id_field

        raw_final_columns = normalized.get('final_columns')
        if not isinstance(raw_final_columns, list):
            raise ValueError('schema.final_columns должен быть списком')

        final_columns: list[str] = []
        seen_columns: set[str] = set()
        for column in raw_final_columns:
            if not isinstance(column, str):
                continue
            clean_column = column.strip()
            if not clean_column or clean_column in seen_columns:
                continue
            seen_columns.add(clean_column)
            final_columns.append(clean_column)
        if id_field not in seen_columns:
            final_columns.insert(0, id_field)
        if not final_columns:
            raise ValueError('schema.final_columns должен быть непустым списком')
        normalized['final_columns'] = final_columns

        column_types = normalized.get('column_types')
        if not isinstance(column_types, dict):
            column_types = {}
        column_types = {str(k).strip(): v for k, v in column_types.items() if str(k).strip()}
        if id_field not in column_types:
            column_types[id_field] = 'id'
        normalized['column_types'] = column_types

        display_names = normalized.get('display_names')
        if not isinstance(display_names, dict):
            display_names = {}
        display_names = {str(k).strip(): str(v).strip() for k, v in display_names.items() if str(k).strip() and str(v).strip()}

        codebook = normalized.get('codebook')
        if not isinstance(codebook, dict):
            codebook = {}
        codebook = {str(k).strip(): v for k, v in codebook.items() if str(k).strip()}

        extraction_logic = normalized.get('extraction_logic')
        if not isinstance(extraction_logic, dict):
            extraction_logic = {}
        extraction_logic = {str(k).strip(): str(v).strip() for k, v in extraction_logic.items() if str(k).strip() and str(v).strip()}

        for column in final_columns:
            if column not in display_names:
                display_names[column] = cls._default_display_name(column)
            if column not in codebook:
                codebook[column] = {}
            if column not in extraction_logic:
                extraction_logic[column] = cls._default_extraction_logic(
                    column=column,
                    column_type=column_types.get(column),
                    display_name=display_names[column],
                    codebook=codebook.get(column),
                )

        normalized['display_names'] = display_names
        normalized['codebook'] = codebook
        normalized['extraction_logic'] = extraction_logic

        drop_source_fields = normalized.get('drop_source_fields')
        if not isinstance(drop_source_fields, list):
            drop_source_fields = []
        normalized['drop_source_fields'] = [str(v).strip() for v in drop_source_fields if str(v).strip()]
        return normalized

    @staticmethod
    def _default_display_name(column: str) -> str:
        return column.replace('_', ' ').strip() or column

    @classmethod
    def _default_extraction_logic(
        cls,
        column: str,
        column_type: str | None,
        display_name: str,
        codebook: Any,
    ) -> str:
        if column_type == 'id':
            return 'Перенести исходный идентификатор записи без изменений.'
        if column_type == 'binary':
            return (
                f"Определить признак '{display_name}' по доступным полям записи и тексту. "
                'Вернуть 1, если признак подтверждён; 0, если он явно отрицается; иначе NULL.'
            )
        if column_type == 'numeric':
            return (
                f"Извлечь числовое значение признака '{display_name}' из доступных полей записи и текста. "
                'Если информации недостаточно, вернуть NULL.'
            )
        if column_type == 'categorical':
            codebook_hint = ''
            if isinstance(codebook, dict) and codebook:
                codebook_hint = ' Использовать только коды из codebook для этого поля.'
            return (
                f"Определить категорию признака '{display_name}' по доступным полям записи и тексту и вернуть только код категории." 
                f'{codebook_hint} Если информации недостаточно, вернуть NULL.'
            )
        return (
            f"Извлечь и нормализовать значение признака '{display_name}' по доступным полям записи и тексту. "
            'Если информации недостаточно, вернуть NULL.'
        )

    @classmethod
    def _coerce_value(cls, value: Any, column_type: str | None) -> Any:
        value = cls.make_json_serializable(value)
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in cls.NULL_STRINGS:
                return None
            value = stripped

        if column_type == 'binary':
            return cls._coerce_binary(value)
        if column_type == 'numeric':
            return cls._coerce_numeric(value)
        if column_type == 'categorical':
            try:
                numeric_value = cls._coerce_numeric(value)
            except ValueError:
                return value
            return numeric_value if numeric_value is not None else value
        return value

    @classmethod
    def _coerce_binary(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            if value in {0, 1}:
                return int(value)
        if isinstance(value, str):
            lowered = value.strip().lower().replace(',', '.')
            if lowered in {'1', '1.0', 'true', 'yes', 'да'}:
                return 1
            if lowered in {'0', '0.0', 'false', 'no', 'нет'}:
                return 0
        raise ValueError(f'Не удалось привести значение {value!r} к binary')

    @classmethod
    def _coerce_numeric(cls, value: Any) -> int | float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            if pd.isna(value):
                return None
            return int(value) if float(value).is_integer() else float(value)
        if isinstance(value, str):
            normalized = value.strip().replace(' ', '').replace(',', '.')
            if normalized.lower() in cls.NULL_STRINGS:
                return None
            try:
                parsed = float(normalized)
            except ValueError as error:
                raise ValueError(f'Не удалось привести значение {value!r} к numeric') from error
            return int(parsed) if parsed.is_integer() else parsed
        raise ValueError(f'Не удалось привести значение {value!r} к numeric')

    @classmethod
    def _validate_schema(cls, schema_json: dict[str, Any]) -> None:
        required_keys = {'id_field', 'final_columns', 'column_types', 'display_names', 'codebook', 'extraction_logic', 'drop_source_fields'}
        missing = required_keys - set(schema_json.keys())
        if missing:
            raise ValueError(f'В schema отсутствуют обязательные ключи: {sorted(missing)}')

        id_field = schema_json.get('id_field')
        if not isinstance(id_field, str) or not id_field.strip():
            raise ValueError('schema.id_field должен быть непустой строкой')

        final_columns = schema_json.get('final_columns', [])
        if not isinstance(final_columns, list) or not final_columns:
            raise ValueError('schema.final_columns должен быть непустым списком')
        if id_field not in final_columns:
            raise ValueError('schema.final_columns должен содержать id_field')

        column_types = schema_json.get('column_types', {})
        display_names = schema_json.get('display_names', {})
        codebook = schema_json.get('codebook', {})
        extraction_logic = schema_json.get('extraction_logic', {})
        drop_source_fields = schema_json.get('drop_source_fields', [])

        if not isinstance(column_types, dict):
            raise ValueError('schema.column_types должен быть объектом')
        if not isinstance(display_names, dict):
            raise ValueError('schema.display_names должен быть объектом')
        if not isinstance(codebook, dict):
            raise ValueError('schema.codebook должен быть объектом')
        if not isinstance(extraction_logic, dict):
            raise ValueError('schema.extraction_logic должен быть объектом')
        if not isinstance(drop_source_fields, list):
            raise ValueError('schema.drop_source_fields должен быть списком')

        for column in final_columns:
            if column not in column_types:
                raise ValueError(f"Для поля '{column}' отсутствует column_types")
            if column not in display_names:
                raise ValueError(f"Для поля '{column}' отсутствует display_names")
            if column not in extraction_logic:
                raise ValueError(f"Для поля '{column}' отсутствует extraction_logic")
            column_type = column_types[column]
            if column_type not in cls.ALLOWED_COLUMN_TYPES:
                raise ValueError(f"Недопустимый тип поля '{column}': {column_type}")
