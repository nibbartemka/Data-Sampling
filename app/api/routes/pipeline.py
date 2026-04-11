from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.config import settings
from app.models.pipeline import PipelineRunRequest, PipelineRunResponse
from app.services.orchestration.combined_pipeline import CombinedPipelineService
from app.storage.run_store import RunStore


router = APIRouter(prefix='/api/v1', tags=['pipeline'])


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


@router.get('/health')
def healthcheck() -> dict[str, str]:
    return {'status': 'ok'}


@router.post('/pipeline/process', response_model=PipelineRunResponse)
async def process_pipeline(
    file: UploadFile = File(...),
    id_field: str = Form(default=settings.ANALYZE.DEFAULT_ID_FIELD),
    text_field: str = Form(default=settings.ANALYZE.DEFAULT_TEXT_FIELD),
    stratify_cols: str = Form(default=','.join(settings.SAMPLING.DEFAULT_STRATIFY_COLS)),
    extra_cols: str = Form(default=','.join(settings.SAMPLING.DEFAULT_EXTRA_COLS)),
    max_features: int = Form(default=settings.ANALYZE.DEFAULT_MAX_FEATURES),
    batch_size: int = Form(default=settings.ANALYZE.DEFAULT_BATCH_SIZE),
    schema_max_tokens: int = Form(default=settings.ANALYZE.DEFAULT_SCHEMA_MAX_TOKENS),
    batch_max_tokens: int = Form(default=settings.ANALYZE.DEFAULT_BATCH_MAX_TOKENS),
    temperature: float = Form(default=settings.ANALYZE.DEFAULT_TEMPERATURE),
    domain_description: str | None = Form(default=None),
) -> PipelineRunResponse:
    suffix = Path(file.filename or '').suffix.lower()
    if suffix != '.xlsx':
        raise HTTPException(status_code=400, detail='Поддерживаются только Excel-файлы .xlsx')

    try:
        request = PipelineRunRequest(
            id_field=id_field,
            text_field=text_field,
            stratify_cols=_parse_csv_list(stratify_cols),
            extra_cols=_parse_csv_list(extra_cols),
            max_features=max_features,
            batch_size=batch_size,
            schema_max_tokens=schema_max_tokens,
            batch_max_tokens=batch_max_tokens,
            temperature=temperature,
            domain_description=domain_description,
        )
        payload = await file.read()
        service = CombinedPipelineService()
        result = service.process_excel(
            input_bytes=payload,
            request=request,
            original_filename=file.filename or 'input.xlsx',
        )
        return PipelineRunResponse(request=request, **result)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f'Внутренняя ошибка pipeline: {error}') from error


@router.get('/runs/{run_id}/artifacts/{artifact_name}')
def download_artifact(run_id: str, artifact_name: str):
    try:
        store = RunStore()
        path = store.resolve_artifact(run_id, artifact_name)
        media_type = 'application/octet-stream'
        if path.suffix.lower() == '.json':
            media_type = 'application/json'
        elif path.suffix.lower() == '.xlsx':
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        return FileResponse(path=path, filename=path.name, media_type=media_type)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
