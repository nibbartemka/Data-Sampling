from __future__ import annotations

import re

import pandas as pd

from app.services.preprocessing.interfaces import BaseRuleEnricher


class PatternRuleEnricher(BaseRuleEnricher):
    PATTERN_MAP = {
        'negation': re.compile(r'\b(?:не|нет|отрицает|не выявлено|не отмечает|отсутствует)\b', re.I),
        'duration': re.compile(r'\b(?:\d+\s*(?:дн|день|дня|нед|недель|недели|мес|месяц|месяцев|лет|года|год)|с \d{1,2}[./]\d{1,2}[./]\d{2,4}|с \d{4} ?г)\b', re.I),
        'dosage': re.compile(r'\b\d+(?:[.,]\d+)?\s*(?:мг|мл|г|мкг|ед|ме|мг/сут|мл/сут)\b', re.I),
        'operation': re.compile(r'\b(?:операц|оперативн|кесарев|хирургическ)\w*', re.I),
        'hospitalization': re.compile(r'\b(?:госпитализ|стационар|выписан|поступил|госпитализац)\w*', re.I),
        'infection_comorbidity': re.compile(r'\b(?:вич|гепатит|туберкул|бактериур|аллерг|гипер|аритм|инфаркт)\w*', re.I),
    }

    def enrich(self, dataset_stratum: pd.DataFrame, text_column: str, record_key_column: str, max_per_pattern: int) -> dict[str, dict]:
        selected: dict[str, dict] = {}
        for pattern_name, regex in self.PATTERN_MAP.items():
            matched = dataset_stratum[dataset_stratum[text_column].str.contains(regex, na=False)].copy()
            if matched.empty:
                continue
            matched['_tmp_len'] = matched[text_column].str.len()
            matched = matched.sort_values('_tmp_len', ascending=False).head(max_per_pattern)
            for _, row in matched.iterrows():
                self._append_reason(selected, row.drop(labels=['_tmp_len'], errors='ignore'), record_key_column, pattern_name, float(row['_tmp_len']))
        return selected

    @staticmethod
    def _append_reason(records: dict[str, dict], row: pd.Series, record_key_column: str, reason: str, score: float) -> None:
        rid = str(row[record_key_column])
        if rid not in records:
            record = row.to_dict()
            record['selection_reasons'] = reason
            record['rule_selection_score'] = float(score)
            records[rid] = record
            return
        existing = records[rid]
        reasons = set(str(existing['selection_reasons']).split('; '))
        reasons.add(reason)
        existing['selection_reasons'] = '; '.join(sorted(r for r in reasons if r))
        existing['rule_selection_score'] = max(float(existing.get('rule_selection_score', 0)), float(score))
