from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ConfigDict, field_validator


class PipelineRunRequest(BaseModel):
    id_field: str = Field(default="PersonID_Ref")
    text_field: str = Field(default="PropertyValue")
    stratify_cols: list[str] = Field(default_factory=lambda: ["ServiceName", "PropertyName"])
    extra_cols: list[str] = Field(default_factory=lambda: ["Sex", "AGE", "StartDate", "EndDate", "MCardMKB"])
    max_features: int = Field(default=15, ge=1, le=100)
    batch_size: int = Field(default=10, ge=1, le=500)
    schema_max_tokens: int = Field(default=6000, ge=256, le=32000)
    batch_max_tokens: int = Field(default=6000, ge=256, le=32000)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)
    domain_description: str | None = Field(default=None)

    @field_validator("id_field", "text_field")
    @classmethod
    def validate_required_field_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Имя колонки не может быть пустым")
        return normalized

    @field_validator("stratify_cols", "extra_cols")
    @classmethod
    def normalize_column_list(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        return normalized

    @field_validator("stratify_cols")
    @classmethod
    def validate_stratify_cols(cls, values: list[str]) -> list[str]:
        if not values:
            raise ValueError("Нужна хотя бы одна колонка для стратификации")
        return values


class PipelineRunResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str
    request: PipelineRunRequest
    input_rows: int
    sample_rows: int
    output_rows: int
    extracted_schema: dict[str, Any] = Field(alias="schema")
    rows: list[dict[str, Any]]
    artifacts: dict[str, str]
