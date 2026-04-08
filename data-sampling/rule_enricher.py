import re
import pandas as pd

from .interfaces import BaseRuleEnricher


class PatternRuleEnricher(BaseRuleEnricher):
    PATTERN_MAP = {
        "negation": re.compile(r"\b(не\b|нет\b|отрицает\b|не выявлено\b|не отмечает\b|отсутствует\b)", re.I),
        "duration": re.compile(r"\b(\d+\s*(дн|день|дня|нед|недель|недели|мес|месяц|месяцев|лет|года|год)|с \d{1,2}[./]\d{1,2}[./]\d{2,4}|с \d{4} ?г)\b", re.I),
        "dosage": re.compile(r"\b\d+([.,]\d+)?\s*(мг|мл|г|мкг|ед|ме|мг/сут|мл/сут)\b", re.I),
        "operation": re.compile(r"\b(операц|оперативн|кесарев|хирургическ)\w*", re.I),
        "hospitalization": re.compile(r"\b(госпитализ|стационар|выписан|поступил|госпитализац)\w*", re.I),
        "infection_comorbidity": re.compile(
            r"\b(вич|гепатит|туберкул|бактериур|аллерг|гипер|аритм|инфаркт)\w*",
            re.I,
        ),
    }

    def enrich(
        self,
        dataset_stratum: pd.DataFrame,
        text_column: str,
        id_column: str,
        max_per_pattern: int
    ) -> dict:
        selected = {}

        for pattern_name, regex in self.PATTERN_MAP.items():
            matched = dataset_stratum[dataset_stratum[text_column].str.contains(regex, na=False)].copy()
            if matched.empty:
                continue

            matched["tmp_len"] = matched[text_column].str.len()
            matched = matched.sort_values("tmp_len", ascending=False).head(max_per_pattern)

            for _, row in matched.iterrows():
                self._append_reason(selected, row, id_column, pattern_name, float(row["tmp_len"]))

        return selected

    def _append_reason(self, records: dict, row: pd.Series, id_column: str, reason: str, score: float) -> None:
        rid = str(row[id_column])

        if rid not in records:
            record = row.to_dict()
            record["selection_reasons"] = reason
            record["selection_score"] = float(score)
            records[rid] = record
            return

        existing = records[rid]
        reasons = set(str(existing["selection_reasons"]).split("; "))
        reasons.add(reason)
        existing["selection_reasons"] = "; ".join(sorted(r for r in reasons if r))
        existing["selection_score"] = max(float(existing["selection_score"]), float(score))
