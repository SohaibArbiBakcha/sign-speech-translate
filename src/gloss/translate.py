"""English text -> approximate ASL gloss.

This is a rule-based heuristic, not a trained model: ASL grammar differs
from English (different word order, no articles/copula), and a fully
correct gloss translation is a research problem in its own right. What we
approximate here, on top of dropping function words English has but ASL
doesn't (articles, auxiliary "be", infinitive "to"):

- **Time topicalization**: ASL conventionally states time context first
  ("YESTERDAY I GO STORE" rather than English's "I went to the store
  yesterday"), so time words are moved to the front of the gloss sequence.
- **Negation at the end**: ASL commonly places negation after what's being
  negated rather than before it, so "not"/"never" etc. move to the end.

This is still a simplified approximation, not real ASL grammar (no
classifiers, non-manual markers, spatial referencing, or verb agreement) —
good enough to drive a gloss -> clip dictionary lookup, not a linguistic
claim.
"""
import spacy

_nlp = None

# POS tags dropped because they mark grammar English has but ASL doesn't
# (articles, copula/auxiliary verbs, infinitive "to").
_DROP_POS = {"DET", "AUX", "PART", "PUNCT"}

_TIME_WORDS = {
    "now", "today", "tomorrow", "yesterday", "before", "after", "later",
    "soon", "year", "month", "week", "day", "morning", "afternoon",
    "evening", "night", "always", "often", "sometimes",
}
_NEGATION_WORDS = {"not", "no", "never", "n't"}


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def text_to_gloss(text: str) -> list[str]:
    """Return a list of gloss tokens approximating ASL word order, e.g.
    'I didn't go to the store yesterday' -> ['YESTERDAY', 'I', 'GO', 'STORE', 'NOT']."""
    doc = _get_nlp()(text)
    lemmas = [
        tok.lemma_.lower()
        for tok in doc
        if tok.pos_ not in _DROP_POS and not tok.is_space
    ]

    time_words = [w for w in lemmas if w in _TIME_WORDS]
    negation_words = [w for w in lemmas if w in _NEGATION_WORDS]
    other_words = [w for w in lemmas if w not in _TIME_WORDS and w not in _NEGATION_WORDS]

    return [w.upper() for w in time_words + other_words + negation_words]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    args = parser.parse_args()
    print(" ".join(text_to_gloss(args.text)))
