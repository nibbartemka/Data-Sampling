from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.routes.pipeline import router as pipeline_router
from app.core.config import ENV_FILES, get_yandex_diagnostics, settings
from app.core.logging import configure_logging


configure_logging()
logger = logging.getLogger(__name__)
app = FastAPI(
    title='LLM Dataset Service',
    version='0.1.0',
    description='Объединённый FastAPI-сервис для sampling и LLM feature extraction',
)
app.include_router(pipeline_router)


@app.on_event('startup')
def log_startup_configuration() -> None:
    env_candidates = [str(path) for path in ENV_FILES]
    diagnostics = get_yandex_diagnostics()
    logger.info(
        'Startup config: env_files=%s, yandex_api_url_set=%s, yandex_api_url_source=%s, yandex_auth_key_set=%s, yandex_auth_key_source=%s, yandex_model_package_set=%s, yandex_model_package_source=%s, yandex_model_name=%s, yandex_auth_type=%s',
        env_candidates,
        diagnostics['api_url_set'],
        diagnostics['api_url_source'],
        diagnostics['auth_key_set'],
        diagnostics['auth_key_source'],
        diagnostics['model_package_set'],
        diagnostics['model_package_source'],
        diagnostics['model_name'],
        diagnostics['auth_type'],
    )


@app.get('/', include_in_schema=False)
def root_redirect() -> RedirectResponse:
    return RedirectResponse(url='/docs')
