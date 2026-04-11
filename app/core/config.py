from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILES: tuple[Path, ...] = (
    PROJECT_ROOT / '.env',
    PROJECT_ROOT / '.env.local',
)


class AppServerSettings(BaseModel):
    HOST: str = '0.0.0.0'
    PORT: int = 8000
    DEBUG: bool = False


class AppStorageSettings(BaseModel):
    ROOT: Path = Field(default_factory=lambda: PROJECT_ROOT)
    RUNS_DIRNAME: str = 'runs'

    @property
    def RUNS_DIR(self) -> Path:
        return self.ROOT / self.RUNS_DIRNAME


class YandexGPTSettings(BaseModel):
    API_URL: str = ''
    AUTH_KEY: str = ''
    AUTH_TYPE: str = 'auto'
    MODEL_PACKAGE: str = ''
    MODEL_NAME: str = 'yandexgpt-lite'


class AnalyzeSettings(BaseModel):
    DEFAULT_ID_FIELD: str = 'PersonID_Ref'
    DEFAULT_TEXT_FIELD: str = 'PropertyValue'
    DEFAULT_MAX_FEATURES: int = 15
    DEFAULT_BATCH_SIZE: int = 10
    DEFAULT_SCHEMA_MAX_TOKENS: int = 6000
    DEFAULT_BATCH_MAX_TOKENS: int = 6000
    DEFAULT_TEMPERATURE: float = 0.1


class SamplingSettings(BaseModel):
    DEFAULT_STRATIFY_COLS: list[str] = ['ServiceName', 'PropertyName']
    DEFAULT_EXTRA_COLS: list[str] = ['Sex', 'AGE', 'StartDate', 'EndDate', 'MCardMKB']
    MIN_TEXT_LENGTH: int = 15
    EMBEDDING_MODEL_NAME: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    DEDUP_SIMILARITY_THRESHOLD: float = 0.92
    DEFAULT_CLUSTERS: int = 5
    SEMANTIC_CENTER_PER_CLUSTER: int = 1
    SEMANTIC_OUTLIER_PER_CLUSTER: int = 1
    MAX_ROWS_PER_STRATUM: int = 30
    TARGETED_MAX_PER_PATTERN_PER_STRATUM: int = 2
    MAX_TOTAL_SAMPLE: int = 300
    MAX_ROWS_PER_GROUP: int = 80


class Settings(BaseSettings):
    APP: AppServerSettings = Field(default_factory=AppServerSettings)
    STORAGE: AppStorageSettings = Field(default_factory=AppStorageSettings)
    YANDEX_GPT: YandexGPTSettings = Field(default_factory=YandexGPTSettings)
    ANALYZE: AnalyzeSettings = Field(default_factory=AnalyzeSettings)
    SAMPLING: SamplingSettings = Field(default_factory=SamplingSettings)

    model_config = SettingsConfigDict(
        env_file=tuple(str(path) for path in ENV_FILES),
        env_nested_delimiter='__',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
    )


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    result: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding='utf-8-sig').splitlines():
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[len('export '):].strip()
            if line.startswith('set '):
                line = line[len('set '):].strip()
            if '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if '#' in value and not (value.startswith('"') or value.startswith("'")):
                value = value.split('#', 1)[0].rstrip()
            value = value.strip().strip('"').strip("'")
            if key:
                result[key] = value
    except Exception:
        return {}
    return result


def _merge_env_files(paths: tuple[Path, ...]) -> tuple[dict[str, str], dict[str, str]]:
    merged: dict[str, str] = {}
    sources: dict[str, str] = {}
    for env_file in paths:
        for key, value in _parse_env_file(env_file).items():
            if value:
                merged[key] = value
                sources[key] = str(env_file)
            elif key not in merged:
                merged[key] = value
                sources[key] = str(env_file)
    return merged, sources


_DOTENV_VALUES, _DOTENV_SOURCES = _merge_env_files(ENV_FILES)


def _first_non_empty(*names: str) -> tuple[str, str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value, f'env:{name}'
        value = _DOTENV_VALUES.get(name)
        if value:
            return value, _DOTENV_SOURCES.get(name, 'dotenv')
    return '', ''


def _detect_auth_type(auth_env_name: str, auth_value: str) -> str:
    upper_name = auth_env_name.upper()
    if 'IAM' in upper_name or 'OAUTH' in upper_name:
        return 'iam_token'
    if auth_value.startswith('t1.'):
        return 'iam_token'
    return 'api_key'


Y_API_NAMES = (
    'YANDEX_GPT__API_URL',
    'YANDEX_GPT_API_URL',
    'YANDEX_API_URL',
    'YC_API_URL',
    'API_URL',
)
Y_AUTH_NAMES = (
    'YANDEX_GPT__AUTH_KEY',
    'YANDEX_GPT__API_KEY',
    'YANDEX_GPT__IAM_TOKEN',
    'YANDEX_GPT__OAUTH_TOKEN',
    'YANDEX_GPT_AUTH_KEY',
    'YANDEX_GPT_API_KEY',
    'YANDEX_GPT_IAM_TOKEN',
    'YANDEX_GPT_OAUTH_TOKEN',
    'YANDEX_API_KEY',
    'YANDEX_IAM_TOKEN',
    'YANDEX_OAUTH_TOKEN',
    'YC_API_KEY',
    'YC_IAM_TOKEN',
    'YC_OAUTH_TOKEN',
    'API_KEY',
    'IAM_TOKEN',
    'OAUTH_TOKEN',
    'AUTH_KEY',
)
Y_AUTH_TYPE_NAMES = (
    'YANDEX_GPT__AUTH_TYPE',
    'YANDEX_GPT_AUTH_TYPE',
    'AUTH_TYPE',
)
Y_PACKAGE_NAMES = (
    'YANDEX_GPT__MODEL_PACKAGE',
    'YANDEX_GPT__FOLDER_ID',
    'YANDEX_GPT__CATALOG_ID',
    'YANDEX_GPT_MODEL_PACKAGE',
    'YANDEX_GPT_FOLDER_ID',
    'YANDEX_GPT_CATALOG_ID',
    'YC_FOLDER_ID',
    'YC_CATALOG_ID',
    'YANDEX_FOLDER_ID',
    'YANDEX_CATALOG_ID',
    'MODEL_PACKAGE',
    'FOLDER_ID',
    'CATALOG_ID',
)
Y_MODEL_NAMES = (
    'YANDEX_GPT__MODEL_NAME',
    'YANDEX_GPT_MODEL_NAME',
    'MODEL_NAME',
)


def _apply_yandex_fallbacks(current: Settings) -> Settings:
    api_url, api_source = _first_non_empty(*Y_API_NAMES)
    auth_key, auth_source = _first_non_empty(*Y_AUTH_NAMES)
    auth_type, _ = _first_non_empty(*Y_AUTH_TYPE_NAMES)
    model_package, package_source = _first_non_empty(*Y_PACKAGE_NAMES)
    model_name, _ = _first_non_empty(*Y_MODEL_NAMES)

    current.YANDEX_GPT.API_URL = current.YANDEX_GPT.API_URL or api_url
    current.YANDEX_GPT.AUTH_KEY = current.YANDEX_GPT.AUTH_KEY or auth_key
    current.YANDEX_GPT.MODEL_PACKAGE = current.YANDEX_GPT.MODEL_PACKAGE or model_package
    current.YANDEX_GPT.MODEL_NAME = current.YANDEX_GPT.MODEL_NAME or model_name or 'yandexgpt-lite'

    if auth_type:
        current.YANDEX_GPT.AUTH_TYPE = auth_type.strip().lower()
    else:
        current.YANDEX_GPT.AUTH_TYPE = _detect_auth_type(auth_source.split(':', 1)[-1], current.YANDEX_GPT.AUTH_KEY) if current.YANDEX_GPT.AUTH_KEY else 'auto'
    return current


settings = _apply_yandex_fallbacks(Settings())
settings.STORAGE.RUNS_DIR.mkdir(parents=True, exist_ok=True)


def get_yandex_diagnostics() -> dict[str, str | bool]:
    _, api_source = _first_non_empty(*Y_API_NAMES)
    _, auth_source = _first_non_empty(*Y_AUTH_NAMES)
    _, package_source = _first_non_empty(*Y_PACKAGE_NAMES)
    return {
        'api_url_set': bool(settings.YANDEX_GPT.API_URL),
        'api_url_source': api_source,
        'auth_key_set': bool(settings.YANDEX_GPT.AUTH_KEY),
        'auth_key_source': auth_source,
        'model_package_set': bool(settings.YANDEX_GPT.MODEL_PACKAGE),
        'model_package_source': package_source,
        'model_name': settings.YANDEX_GPT.MODEL_NAME,
        'auth_type': settings.YANDEX_GPT.AUTH_TYPE,
    }
