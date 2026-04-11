from __future__ import annotations

import re

import pandas as pd

from app.services.preprocessing.interfaces import BaseTextProcessor


class TextNormalizer(BaseTextProcessor):
    ABBREVIATIONS: dict[str, str] = {
        r'\bГБ\b': 'гипертоническая болезнь',
        r'\bОИМ\b': 'острый инфаркт миокарда',
        r'\bОНМК\b': 'острое нарушение мозгового кровообращения',
        r'\bЭНМГ\b': 'электронейромиография',
        r'\bНСР\b': 'нарушения сердечного ритма',
    }

    def process_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        processed = str(text).strip()
        processed = processed.replace("\xa0", " ")
        processed = processed.replace("ё", "е")
        processed = re.sub(r"\s+", " ", processed)
        processed = processed.lower()

        for abbr, full in self.ABBREVIATIONS.items():
            processed = re.sub(abbr, full, processed, flags=re.IGNORECASE)

        processed = re.sub(r"\s+", " ", processed).strip()
        return processed
