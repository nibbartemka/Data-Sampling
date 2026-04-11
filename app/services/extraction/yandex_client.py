from __future__ import annotations

import json
import logging
from typing import Any

import requests


logger = logging.getLogger(__name__)


class YandexGPTClient:
    def __init__(
        self,
        api_url: str,
        auth_key: str,
        model_package: str,
        model_name: str = 'yandexgpt-lite',
        auth_type: str = 'api_key',
    ) -> None:
        missing: list[str] = []
        if not api_url:
            missing.append('API_URL')
        if not auth_key:
            missing.append('AUTH_KEY')
        if not model_package:
            missing.append('MODEL_PACKAGE')
        if missing:
            raise RuntimeError(
                'Не настроены параметры YandexGPT: '
                + ', '.join(missing)
                + '. Ожидаются переменные окружения вида '
                + 'YANDEX_GPT__API_URL, YANDEX_GPT__AUTH_KEY, YANDEX_GPT__MODEL_PACKAGE '
                + 'или их совместимые алиасы.'
            )
        self._api_url = api_url
        self._auth_key = auth_key
        self._auth_type = (auth_type or 'api_key').lower()
        self._model_uri = f'gpt://{model_package}/{model_name}'

    def call(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 4000,
        temperature: float = 0.0,
        retries: int = 2,
        timeout: int = 180,
    ) -> str:
        headers = {'Content-Type': 'application/json', 'Authorization': self._build_auth_header()}
        body = {
            'modelUri': self._model_uri,
            'completionOptions': {'stream': False, 'temperature': temperature, 'maxTokens': max_tokens},
            'messages': messages,
        }
        last_error: Exception | None = None
        for attempt in range(max(retries, 1)):
            try:
                response = requests.post(self._api_url, headers=headers, json=body, timeout=timeout)
                if response.status_code >= 400:
                    raise requests.HTTPError(f'HTTP {response.status_code}: {response.text}', response=response)
                data = response.json()
                return self._extract_text(data)
            except Exception as error:  # noqa: BLE001
                last_error = error
                logger.error('YandexGPT attempt %s failed: %s', attempt + 1, error)
        raise RuntimeError(f'Не удалось получить ответ от YandexGPT: {last_error}')

    def _build_auth_header(self) -> str:
        if self._auth_type in {'iam', 'iam_token', 'oauth', 'oauth_token', 'bearer'}:
            return f'Bearer {self._auth_key}'
        return f'Api-Key {self._auth_key}'

    @staticmethod
    def _extract_text(response: Any) -> str:
        alternatives = response.get('result', {}).get('alternatives', [])
        if not alternatives:
            logger.debug('Unexpected response: %s', json.dumps(response, ensure_ascii=False))
            raise ValueError('В ответе нет alternatives')
        text = alternatives[0].get('message', {}).get('text', '')
        if not text:
            logger.debug('Unexpected response: %s', json.dumps(response, ensure_ascii=False))
            raise ValueError('В ответе нет текста модели')
        return text
