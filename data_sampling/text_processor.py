import re

import pandas as pd

from .interfaces import BaseTextProcessor


class TextNormalizer(BaseTextProcessor):
    ABBREVIATIONS: dict[str, str] = {
        r'\bГБ\b': 'гипертоническая болезнь',
        r'\bОИМ\b': 'острый инфаркт миокарда',
        r'\bОНМК\b': 'острое нарушение мозгового кровообращения',
        r'\bЭНМГ\b': 'электронейромиография',
        r'\bНСР\b': 'Нарушения сердечного ритма'
    }

    def process_text(self, text: str) -> str:
        if pd.isna(text):
            return ""

        text = str(text).strip()
        text = text.replace("\xa0", " ")
        text = text.replace("ё", "е")
        text = re.sub(r"\s+", " ", text)
        text = text.lower()

        for abbr, full in self.ABBREVIATIONS.items():
            text = re.sub(abbr, full, text, flags=re.IGNORECASE)

        text = re.sub(r"\s+", " ", text).strip()
        return text
