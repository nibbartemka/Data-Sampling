from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil

import pandas as pd

from app.core.config import settings
from app.models.pipeline import PipelineRunRequest
from app.services.extraction.pipeline import FeatureExtractionPipeline
from app.services.extraction.prompt_manager import PromptManager
from app.services.extraction.yandex_client import YandexGPTClient
from app.services.preprocessing.clusterer import KMeansClusterer
from app.services.preprocessing.embedder import SentenceTransformerEmbedder
from app.services.preprocessing.pipeline import SampleBuilderPipeline
from app.services.preprocessing.rule_enricher import PatternRuleEnricher
from app.services.preprocessing.sample_merger import SimpleSampleMerger
from app.services.preprocessing.semantic_selector import SimpleSemanticSelector
from app.services.preprocessing.stratifier import SimpleStratifier
from app.services.preprocessing.text_processor import TextNormalizer
from app.storage.run_store import RunStore


class CombinedPipelineService:
    def __init__(self) -> None:
        self.run_store = RunStore()
        self.sample_pipeline = SampleBuilderPipeline(
            text_processor=TextNormalizer(),
            stratifier=SimpleStratifier(),
            embedder=SentenceTransformerEmbedder(settings.SAMPLING.EMBEDDING_MODEL_NAME),
            clusterer=KMeansClusterer(),
            rule_enricher=PatternRuleEnricher(),
            semantic_selector=SimpleSemanticSelector(),
            sample_merger=SimpleSampleMerger(),
        )

    def process_excel(
        self,
        input_bytes: bytes,
        request: PipelineRunRequest,
        original_filename: str = 'input.xlsx',
    ) -> dict[str, Any]:
        run_id, run_dir = self.run_store.create_run_dir()
        try:
            input_path = self.run_store.save_upload(run_dir, original_filename, input_bytes)
            try:
                input_df = pd.read_excel(input_path)
            except Exception as error:  # noqa: BLE001
                raise ValueError('Не удалось прочитать Excel-файл. Убедитесь, что загружен корректный .xlsx') from error
            if input_df.empty:
                raise ValueError('Загруженный Excel-файл не содержит строк для обработки')

            sample_df = self.sample_pipeline.build_sample(input_df=input_df, request=request)
            sample_path = self.run_store.save_dataframe(run_dir, 'sample.xlsx', sample_df)

            extraction_sample_df = sample_df.drop(
                columns=['stratum_key', 'selection_reasons', 'semantic_selection_score', 'rule_selection_score'],
                errors='ignore',
            )
            sample_records = FeatureExtractionPipeline.make_json_serializable(extraction_sample_df.to_dict(orient='records'))
            raw_records = FeatureExtractionPipeline.make_json_serializable(input_df.to_dict(orient='records'))

            schema_prompt_path = Path(__file__).resolve().parent.parent / 'extraction' / 'prompts' / 'build_schema_prompt.md'
            batch_prompt_path = Path(__file__).resolve().parent.parent / 'extraction' / 'prompts' / 'build_batch_prompt.md'
            schema_prompt = PromptManager.build_prompt(
                schema_prompt_path,
                max_features=request.max_features,
                id_field=request.id_field,
                text_field=request.text_field,
            )
            batch_prompt = PromptManager.build_prompt(
                batch_prompt_path,
                max_features=request.max_features,
                id_field=request.id_field,
                text_field=request.text_field,
            )

            yandex_client = YandexGPTClient(
                api_url=settings.YANDEX_GPT.API_URL,
                auth_key=settings.YANDEX_GPT.AUTH_KEY,
                model_package=settings.YANDEX_GPT.MODEL_PACKAGE,
                model_name=settings.YANDEX_GPT.MODEL_NAME,
                auth_type=settings.YANDEX_GPT.AUTH_TYPE,
            )
            extraction_pipeline = FeatureExtractionPipeline(yandex_client, schema_prompt, batch_prompt)
            extraction_result = extraction_pipeline.run(
                sample_records=sample_records,
                raw_records=raw_records,
                batch_size=request.batch_size,
                schema_max_tokens=request.schema_max_tokens,
                batch_max_tokens=request.batch_max_tokens,
                temperature=request.temperature,
                domain_description=request.domain_description,
            )

            schema = extraction_result['schema']
            rows = extraction_result['rows']
            output_df = FeatureExtractionPipeline.apply_schema_to_dataframe(rows, schema=schema)

            schema_path = self.run_store.save_json(run_dir, 'schema.json', schema)
            output_json_path = self.run_store.save_json(run_dir, 'output.json', rows)
            output_excel_path = self.run_store.save_dataframe(run_dir, 'output.xlsx', output_df)

            return {
                'run_id': run_id,
                'input_rows': len(input_df),
                'sample_rows': len(sample_df),
                'output_rows': len(rows),
                'schema': schema,
                'rows': rows,
                'artifacts': {
                    'input': f'/api/v1/runs/{run_id}/artifacts/{input_path.name}',
                    'sample': f'/api/v1/runs/{run_id}/artifacts/{sample_path.name}',
                    'schema': f'/api/v1/runs/{run_id}/artifacts/{schema_path.name}',
                    'output_json': f'/api/v1/runs/{run_id}/artifacts/{output_json_path.name}',
                    'output_excel': f'/api/v1/runs/{run_id}/artifacts/{output_excel_path.name}',
                },
            }
        except Exception:
            shutil.rmtree(run_dir, ignore_errors=True)
            raise
