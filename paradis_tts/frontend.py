"""Offline French frontend matching the training normalizer and phoneme IDs."""
from dataclasses import dataclass
import re

from paradis_tts.normalization import normalize_sentence


@dataclass(frozen=True)
class TextChunk:
    text: str
    phonemes: str
    ids: tuple[int, ...]


class FrenchFrontend:
    def __init__(self, vocab, *, max_phonemes=400, g2p=None):
        if not vocab or any(not isinstance(k, str) or len(k) != 1 or
                            not isinstance(v, int) or not 1 <= v < 178 for k, v in vocab.items()):
            raise ValueError('Invalid training phoneme vocabulary')
        if not 1 <= max_phonemes <= 400:
            raise ValueError('max_phonemes must be between 1 and 400')
        self.vocab, self.max_phonemes = dict(vocab), max_phonemes
        if g2p is None:
            from misaki.espeak import EspeakG2P
            g2p = EspeakG2P(language='fr-fr')
        self.g2p = g2p

    def prepare(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Entrer un texte français non vide.')
        if len(text) > 10000:
            raise ValueError('Texte limité à 10 000 caractères par appel.')
        normalized, rules = normalize_sentence(text)
        if not normalized or not any(c.isalnum() for c in normalized):
            raise ValueError('Le texte ne contient aucun mot à prononcer.')
        # Preserve complete training-sized utterances, including initials and punctuation.
        # Split only when the actual phoneme sequence exceeds the model's budget.
        pending = [normalized]
        chunks = []
        while pending:
            part = pending.pop().strip()
            if not part:
                continue
            phonemes, _ = self.g2p(part)
            if not phonemes:
                raise ValueError(f'Aucun phonème produit pour : {part}')
            unknown = sorted(set(phonemes)-self.vocab.keys())
            if unknown:
                raise ValueError(f'Phonèmes absents du vocabulaire entraîné : {unknown!r}')
            if len(phonemes) > self.max_phonemes:
                boundaries = [m.start() for m in re.finditer(r'\s+', part)]
                sentences = [m.start() for m in re.finditer(r'(?<=[.!?…])\s+', part)
                             if not re.search(r'\b[A-ZÀ-Ý]\.$', part[:m.start()])]
                boundaries = sentences or boundaries
                if not boundaries:
                    raise ValueError('Un mot dépasse la limite de phonèmes ; le texte ne sera pas tronqué.')
                cut = min(boundaries, key=lambda i: abs(i-len(part)/2))
                pending.extend([part[cut:].strip(), part[:cut].strip()])
                continue
            chunks.append(TextChunk(part, phonemes, (0, *[self.vocab[c] for c in phonemes], 0)))
            if len(chunks) > 64:
                raise ValueError('Texte limité à 64 segments par appel.')
        return normalized, rules, chunks
