from data_sampling.pipeline import Pipeline
from data_sampling.clusterer import KMeansClusterer
from data_sampling.data_manager import ExcelDataManager
from data_sampling.embedder import SentenceTransformerEmbedder
from data_sampling.rule_enricher import PatternRuleEnricher
from data_sampling.semantic_selector import SimpleSemanticSelector
from data_sampling.stratifier import SimpleStratifier
from data_sampling.text_processor import TextNormalizer
from data_sampling.sample_merger import SimpleSampleMerger
from config import settings


if __name__ == '__main__':
    pipeline = Pipeline(
        data_manager=ExcelDataManager(),
        text_processor=TextNormalizer(),
        stratifier=SimpleStratifier(),
        embedder=SentenceTransformerEmbedder(settings.EMBEDDING.MODEL_NAME),
        clusterer=KMeansClusterer(),
        rule_enricher=PatternRuleEnricher(),
        semantic_selector=SimpleSemanticSelector(),
        sample_merger=SimpleSampleMerger(),
    )

    pipeline.run()
