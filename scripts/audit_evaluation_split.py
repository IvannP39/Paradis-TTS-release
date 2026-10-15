"""Recheck article separation and exact/near-duplicate exclusion using public metadata."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
WORD = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[,.]\d+)?",re.UNICODE)


def canonical(text):
    return ' '.join(word.casefold() for word in WORD.findall(unicodedata.normalize('NFC',text)))


def main():
    prior = [json.loads(line) for line in (ROOT/'provenance/training_text_attribution.jsonl').read_text(encoding='utf-8').splitlines()]
    heldout = [json.loads(line) for line in (ROOT/'evaluation/heldout_v0.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = {row['source_group'] for row in prior}
    exact = {canonical(row['normalized_text']) for row in prior}
    index = defaultdict(set)
    sets = []

    def shingles(text):
        words = canonical(text).split()
        return set(zip(words,words[1:],words[2:]))

    def add(entry):
        for token in entry:
            index[token].add(len(sets))
        sets.append(entry)

    for row in prior:
        add(shingles(row['normalized_text']))
    for row in heldout:
        assert row['source_group'] not in groups
        assert canonical(row['normalized_text']) not in exact
        entry = shingles(row['normalized_text'])
        candidates = set().union(*(index[token] for token in entry))
        assert not any(len(entry & sets[i])/len(entry | sets[i]) >= .85 for i in candidates)
        add(entry)
        groups.add(row['source_group'])
        exact.add(canonical(row['normalized_text']))
    assert len(heldout)==300 and len(prior)==13000
    print(json.dumps({'status':'HELDOUT_AUDIT_PASS','prior_sentences':len(prior),
        'heldout_sentences':len(heldout),'article_overlap':0,'exact_overlap':0,
        'near_overlap_at_jaccard_085':0}))


if __name__ == '__main__':
    main()
