from pydantic import BaseModel
from pydantic_settings import BaseSettings


class LocalRepositoryConfig(BaseModel):
    INPUT_FILE_PATH: str = "input.xlsx"
    OUTPUT_FILE_PATH: str = "ouput.xlsx"


class TextProcessingConfig(BaseModel):
    MIN_TEXT_LENGTH: int = 15


class EmbeddingConfig(BaseModel):
    MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DEDUP_SIMILARITY_THRESHOLD: float = 0.92


class ClusteringConfig(BaseModel):
    DEFAULT_CLUSTERS: int = 5
    SEMANTIC_CENTER_PER_CLUSTER: int = 1
    SEMANTIC_OUTLIER_PER_CLUSTER: int = 1


class StratificationConfig(BaseModel):
    MAX_ROWS_PER_STRATUM: int = 30
    TARGETED_MAX_PER_PATTERN_PER_STRATUM: int = 2
    RARE_CASES_MAX_PER_STRATUM: int = 3


class SamplingLimitsConfig(BaseModel):
    MAX_TOTAL_SAMPLE: int = 300
    MAX_ROWS_PER_SERVICE: int = 80


class RareScorerConfig(BaseModel):
    RARE_TERM_MIN_FREQ: int = 2
    RARE_TERM_MAX_FREQ_RATIO: float = 0.03


class ColumnMappingConfig(BaseModel):
    ID_COL: str = "PersonID_Ref"
    TEXT_COL: str = "PropertyValue"
    STRATIFY_COLS: list[str] = ["ServiceName",
                                "PropertyName",]
    EXTRA_COLS: list[str] = [
        "Sex",
        "AGE",
        "StartDate",
        "EndDate",
        "MCardMKB",
    ]

    @property
    def REQUIRED_COLS(self) -> list[str]:
        return [
            self.ID_COL,
            self.TEXT_COL,
            self.STRATIFY_COLS,
            *self.EXTRA_COLS
        ]


class Settings(BaseSettings):
    LOCAL_REP: LocalRepositoryConfig
    COLS_MAP: ColumnMappingConfig 
    TEXT_PROC: TextProcessingConfig
    EMBEDDING: EmbeddingConfig
    CLUSTERING: ClusteringConfig
    STRATIFY: StratificationConfig
    RARE_SCORER: RareScorerConfig
    SAMPLING: SamplingLimitsConfig