import pandas as pd

from config import settings
from .interfaces import (
    BaseClusterer,
    BaseDataManager,
    BaseEmbedder,
    BaseRuleEnricher,
    BaseSampleMerger,
    BaseSemanticSelector,
    BaseStratifier,
    BaseTextProcessor,
)


__all__ = [
    'Pipeline',
]


class Pipeline:
    def __init__(
        self,
        data_manager: BaseDataManager,
        text_processor: BaseTextProcessor,
        stratifier: BaseStratifier,
        embedder: BaseEmbedder,
        clusterer: BaseClusterer,
        rule_enricher: BaseRuleEnricher,
        semantic_selector: BaseSemanticSelector,
        sample_merger: BaseSampleMerger,
    ):
        self.data_manager = data_manager
        self.text_processor = text_processor
        self.stratifier = stratifier
        self.embedder = embedder
        self.clusterer = clusterer
        self.semantic_selector = semantic_selector
        self.rule_enricher = rule_enricher
        self.sample_merger = sample_merger

    def build_sample(self) -> pd.DataFrame:
        df = self.data_manager.load_dataset(
            settings.LOCAL_REP.INPUT_FILE_PATH,
            settings.COLS_MAP.REQUIRED_COLS,
        )

        df = self.text_processor.preprocess(
            df=df,
            text_column=settings.COLS_MAP.TEXT_COL,
            min_text_len=settings.TEXT_PROC.MIN_TEXT_LENGTH,
        )

        df = self.stratifier.build_stratum_key(
            df=df,
            stratify_columns=settings.COLS_MAP.STRATIFY_COLS,
        )

        parts = []

        for stratum_key, df_stratum in self.stratifier.split(df):
            texts = df_stratum[settings.COLS_MAP.TEXT_COL].tolist()
            embeddings = self.embedder.encode(texts)

            cluster_labels = self.clusterer.cluster(
                embeddings,
                settings.CLUSTERING.DEFAULT_CLUSTERS,
            )

            semantic_records = self.semantic_selector.select(
                df_stratum=df_stratum,
                embeddings=embeddings,
                cluster_labels=cluster_labels,
                id_column=settings.COLS_MAP.ID_COL,
                center_per_cluster=settings.CLUSTERING.SEMANTIC_CENTER_PER_CLUSTER,
                outlier_per_cluster=settings.CLUSTERING.SEMANTIC_OUTLIER_PER_CLUSTER,
            )

            rule_records = self.rule_enricher.enrich(
                df_stratum=df_stratum,
                text_column=settings.COLS_MAP.TEXT_COL,
                id_column=settings.COLS_MAP.ID_COL,
                max_per_pattern=settings.STRATIFY.TARGETED_MAX_PER_PATTERN_PER_STRATUM,
            )

            merged = self.sample_merger.merge_record_dicts(
                semantic_records,
                rule_records,
            )

            if merged.empty:
                continue

            merged["stratum_key"] = stratum_key
            merged = merged.sort_values(
                "selection_score",
                ascending=False,
            ).head(settings.STRATIFY.MAX_ROWS_PER_STRATUM)

            parts.append(merged)

        if not parts:
            return pd.DataFrame()

        result = pd.concat(parts, ignore_index=True)

        # Ограничение по группам.
        # По текущей логике используем первую колонку стратификации как верхнеуровневую группу,
        # например ServiceName.
        group_col = settings.COLS_MAP.STRATIFY_COLS[0]

        result = self.sample_merger.limit_per_group(
            result,
            group_col,
            max_rows_per_group=settings.SAMPLING.MAX_ROWS_PER_GROUP,
        )

        result = self.sample_merger.deduplicate(
            df=result,
            text_column=settings.COLS_MAP.TEXT_COL,
            similarity_threshold=settings.EMBEDDING.DEDUP_SIMILARITY_THRESHOLD,
            embedder=self.embedder,
        )

        result["reason_count"] = result["selection_reasons"].map(
            lambda x: len(str(x).split("; "))
        )

        result = result.sort_values(
            by=["reason_count", "selection_score"],
            ascending=[False, False],
        ).head(settings.SAMPLING.MAX_TOTAL_SAMPLE).copy()

        result = result.drop(columns=["reason_count"], errors="ignore").reset_index(drop=True)
        return result

    def run(self) -> pd.DataFrame:
        result = self.build_sample()
        if not result.empty:
            self.data_manager.save_dataset(
                result,
                settings.LOCAL_REP.OUTPUT_FILE_PATH,
            )
        return result
