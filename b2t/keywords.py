from typing import List

import yake


def extract_keywords(
    captions: str,
    deduplication_threshold: float = 0.9,
    max_ngram_size: int = 3,
    num_keywords: int = 20,
    **kwargs,
) -> List[int]:
    language = "en"
    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        n=max_ngram_size,
        dedupLim=deduplication_threshold,
        top=num_keywords,
        features=None,
        **kwargs,
    )

    keywords = custom_kw_extractor.extract_keywords(captions)
    keywords = [keyword[0] for keyword in keywords]
    return keywords
