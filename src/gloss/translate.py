"""English text -> approximate ASL gloss.

This is a rule-based heuristic, not a trained model: ASL grammar differs
from English (different word order, no articles/copula), and a fully
correct gloss translation is a research problem in its own right. What we
do here is a common simplified approximation used in prototype systems:
drop function words that don't carry sign content (articles, auxiliary
"be", infinitive "to"), lemmatize the rest, and upper-case them per ASL
gloss-writing convention.

Good enough to drive a gloss -> clip dictionary lookup; not a substitute
for real ASL grammar.
"""
import spacy

_nlp = None

# POS tags dropped because they mark grammar English has but ASL doesn't
# (articles, copula/auxiliary verbs, infinitive "to").
_DROP_POS = {"DET", "AUX", "PART", "PUNCT"}


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def text_to_gloss(text: str) -> list[str]:
    """Return a list of gloss tokens, e.g. 'Is your mother fine?' -> ['MOTHER', 'FINE']."""
    doc = _get_nlp()(text)
    return [
        tok.lemma_.upper()
        for tok in doc
        if tok.pos_ not in _DROP_POS and not tok.is_space
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("text", type=str)
    args = parser.parse_args()
    print(" ".join(text_to_gloss(args.text)))
