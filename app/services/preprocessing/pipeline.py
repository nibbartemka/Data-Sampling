from __future__ import annotations

import pandas as pd

from app.core.config import settings
from app.models.pipeline import PipelineRunRequest
from app.services.preprocessing.interfaces import (
    BaseClusterer,
    BaseEmbedder,
    BaseRuleEnricher,
    BaseSampleMerger,
    BaseSemanticSelector,
    BaseStratifier,
    BaseTextProcessor,
)


class SampleBuilderPipeline:
    INTERNAL_ROW_KEY = "__sample_row_uid"

    def __init__(
        self,
        text_processor: BaseTextProcessor,
        stratifier: BaseStratifier,
        embedder: BaseEmbedder,
        clusterer: BaseClusterer,
        rule_enricher: BaseRuleEnricher,
        semantic_selector: BaseSemanticSelector,
        sample_merger: BaseSampleMerger,
    ) -> None:
        self.text_processor = text_processor
        self.stratifier = stratifier
        self.embedder = embedder
        self.clusterer = clusterer
        self.semantic_selector = semantic_selector
        self.rule_enricher = rule_enricher
        self.sample_merger = sample_merger

    @staticmethod
    def required_columns(request: PipelineRunRequest) -> list[str]:
        ordered_columns = [request.id_field, request.text_field, *request.stratify_cols, *request.extra_cols]
        result: list[str] = []
        seen: set[str] = set()
        for column in ordered_columns:
            if column in seen:
                continue
            seen.add(column)
            result.append(column)
        return result

    def build_sample(self, input_df: pd.DataFrame, request: PipelineRunRequest) -> pd.DataFrame:
        if input_df.empty:
            raise ValueError('Входной файл пустой — отсутствуют строки для обработки')

        required_columns = self.required_columns(request)
        missing_columns = [col for col in required_columns if col not in input_df.columns]
        if missing_columns:
            raise ValueError(f'Во входном файле отсутствуют обязательные колонки: {missing_columns}')

        df = input_df[required_columns].copy()
        text_col = request.text_field
        df[self.INTERNAL_ROW_KEY] = [str(idx) for idx in range(len(df))]
        df[text_col] = df[text_col].map(self.text_processor.process_text)
        df = df[df[text_col].str.len() >= settings.SAMPLING.MIN_TEXT_LENGTH].copy().reset_index(drop=True)
        if df.empty:
            raise ValueError(
                'После предобработки не осталось ни одной строки: проверьте текстовую колонку и минимальную длину текста'
            )

        df = self.stratifier.build_stratum_key(dataset=df, stratify_columns=request.stratify_cols)
        parts: list[pd.DataFrame] = []

        for stratum_key, df_stratum in self.stratifier.split(df):
            texts = df_stratum[text_col].tolist()
            embeddings = self.embedder.encode(texts)
            cluster_labels = self.clusterer.cluster(embeddings, settings.SAMPLING.DEFAULT_CLUSTERS)
            semantic_records = self.semantic_selector.select(
                dataset_stratum=df_stratum,
                embeddings=embeddings,
                cluster_labels=cluster_labels,
                record_key_column=self.INTERNAL_ROW_KEY,
                center_per_cluster=settings.SAMPLING.SEMANTIC_CENTER_PER_CLUSTER,
                outlier_per_cluster=settings.SAMPLING.SEMANTIC_OUTLIER_PER_CLUSTER,
            )
            rule_records = self.rule_enricher.enrich(
                dataset_stratum=df_stratum,
                text_column=text_col,
                record_key_column=self.INTERNAL_ROW_KEY,
                max_per_pattern=settings.SAMPLING.TARGETED_MAX_PER_PATTERN_PER_STRATUM,
            )
            merged = self.sample_merger.merge_record_dicts(semantic_records, rule_records)
            if merged.empty:
                continue
            merged['stratum_key'] = stratum_key
            merged = merged.sort_values(
                by=['semantic_selection_score', 'rule_selection_score'],
                ascending=[False, False],
            ).head(settings.SAMPLING.MAX_ROWS_PER_STRATUM)
            parts.append(merged)

        if not parts:
            raise ValueError('Не удалось сформировать representative sample по входным данным')

        result = pd.concat(parts, ignore_index=True)
        group_col = request.stratify_cols[0]
        result = self.sample_merger.limit_per_group(
            result,
            group_col,
            max_rows_per_group=settings.SAMPLING.MAX_ROWS_PER_GROUP,
        )
        result = self.sample_merger.deduplicate(
            dataset=result,
            text_column=text_col,
            similarity_threshold=settings.SAMPLING.DEDUP_SIMILARITY_THRESHOLD,
            embedder=self.embedder,
        )
        result['reason_count'] = result['selection_reasons'].map(lambda x: len(str(x).split('; ')))
        result = result.sort_values(
            by=['reason_count', 'semantic_selection_score', 'rule_selection_score'],
            ascending=[False, False, False],
        ).head(settings.SAMPLING.MAX_TOTAL_SAMPLE).copy()
        return result.drop(columns=['reason_count', self.INTERNAL_ROW_KEY], errors='ignore').reset_index(drop=True)
