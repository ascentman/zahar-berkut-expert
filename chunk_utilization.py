from langchain_core.documents import Document

# Threshold: chunk is "utilized" if its score exceeds this value.
# Tuned for Ukrainian text with word unigram matching.
# Filters short/common words to reduce noise.
UTILIZATION_THRESHOLD = 0.12

# Words shorter than this are ignored (filters articles, prepositions, conjunctions)
_MIN_WORD_LEN = 4


def _content_words(text: str) -> set[str]:
    """Extract meaningful words, filtering short/common ones."""
    return {w for w in text.lower().split() if len(w) >= _MIN_WORD_LEN}


def check_utilization(answer: str, chunks: list[Document]) -> list[float]:
    """Score how much of each chunk's key words appear in the answer.

    Uses word unigram overlap on content words (4+ chars).
    Short words (і, що, але, від, це, ...) are excluded to avoid
    false positives from Ukrainian function words.
    """
    answer_words = _content_words(answer)
    if not answer_words:
        return [0.0] * len(chunks)

    scores = []
    for chunk in chunks:
        chunk_words = _content_words(chunk.page_content)
        if not chunk_words:
            scores.append(0.0)
            continue
        overlap = len(answer_words & chunk_words) / len(chunk_words)
        scores.append(round(overlap, 4))
    return scores
