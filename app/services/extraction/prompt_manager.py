from __future__ import annotations

from pathlib import Path
from string import Template


class PromptManager:
    @staticmethod
    def build_prompt(path: str | Path, *args, **kwargs) -> str:
        with open(path, 'r', encoding='utf-8') as fh:
            prompt = fh.read()
        for key, value in kwargs.items():
            prompt = prompt.replace(f'{{{key}}}', str(value))
        template = Template(prompt)
        return template.safe_substitute(*args, **kwargs)
