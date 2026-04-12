from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

import pandas as pd

from app.core.config import settings


class RunStore:
    SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-zА-Яа-я0-9._-]+")
    SAFE_RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

    def create_run_dir(self) -> tuple[str, Path]:
        run_id = uuid4().hex
        run_dir = settings.STORAGE.RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_id, run_dir

    def sanitize_filename(self, filename: str, default: str = 'input.xlsx') -> str:
        base_name = Path(filename or default).name
        cleaned = self.SAFE_FILENAME_PATTERN.sub('_', base_name).strip('._')
        if not cleaned:
            cleaned = default
        return cleaned

    def save_upload(self, run_dir: Path, filename: str, payload: bytes) -> Path:
        safe_name = self.sanitize_filename(filename)
        path = run_dir / safe_name
        path.write_bytes(payload)
        return path

    def save_dataframe(self, run_dir: Path, filename: str, df: pd.DataFrame) -> Path:
        safe_name = self.sanitize_filename(filename, default='data.xlsx')
        path = run_dir / safe_name
        df.to_excel(path, index=False)
        return path

    def save_json(self, run_dir: Path, filename: str, data: object) -> Path:
        safe_name = self.sanitize_filename(filename, default='data.json')
        path = run_dir / safe_name
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        return path

    def resolve_artifact(self, run_id: str, artifact_name: str) -> Path:
        if not self.SAFE_RUN_ID_PATTERN.fullmatch(run_id or ''):
            raise FileNotFoundError(f"Artifact not found: {artifact_name}")
        safe_name = self.sanitize_filename(artifact_name, default='artifact')
        runs_root = settings.STORAGE.RUNS_DIR.resolve()
        run_dir = (runs_root / run_id).resolve()
        path = (run_dir / safe_name).resolve()
        if run_dir.parent != runs_root:
            raise FileNotFoundError(f"Artifact not found: {artifact_name}")
        if runs_root not in path.parents or run_dir not in path.parents or not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Artifact not found: {artifact_name}")
        return path
