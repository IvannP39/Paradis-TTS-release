"""Bounded French inference normalization; original training normalization stays frozen."""
import re
import unicodedata
from paradis_tts.normalization import normalize_sentence as baseline,cardinal

VERSION='fr-runtime-2'
LETTERS=dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ',[
    'a','bé','cé','dé','e','èf','gé','hache','i','ji','ka','èle','ème',
    'ène','o','pé','ku','ère','ès','té','u','vé','double vé','iks','i grec','zède']))
ACRONYMS={name:' '.join(LETTERS[letter] for letter in name)
          for name in ('SNCF','TGV','RATP','EDF','CNRS','USB','UE','IA','TV','GPS')}


def address_words(value):
    replacements={'.':' point ','@':' arobase ','-':' tiret ','_':' tiret bas ',
                  '+':' plus ','/':' barre oblique ',':':' deux points '}
    value=''.join(' '+cardinal(c)+' ' if c.isdigit() else replacements.get(c,c) for c in value)
    # Common suffixes are spelled explicitly; names themselves stay readable words.
    return re.sub(r'\b(fr|com|org|net|eu)\b',lambda m:' '.join(LETTERS[c] for c in m[0].upper()),value,flags=re.I)


def normalize_sentence(text):
    if not isinstance(text,str) or not text.strip(): raise ValueError('Entrer un texte non vide.')
    if len(text)>10000: raise ValueError('Texte limité à 10 000 caractères.')
    text=unicodedata.normalize('NFC',text)
    rules=[]
    def apply(pattern,replacement,name,flags=0):
        nonlocal text
        text,count=re.subn(pattern,replacement,text,flags=flags)
        if count: rules.append(name)
    apply(r'(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+',lambda m:address_words(m[0]),'email')
    def url(match):
        value=match[0];tail=''
        while value and value[-1] in '.,;!?)': tail=value[-1]+tail;value=value[:-1]
        value=re.sub(r'^https?://','',value,flags=re.I)
        return address_words(value)+tail
    apply(r'\b(?:https?://|www\.)[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[A-Za-z0-9_./-]*)?',url,'url',re.I)
    apply(r'\b(?:'+ '|'.join(ACRONYMS)+r')\b',lambda m:ACRONYMS[m[0]],'acronym')
    apply(r'\bMlle\.?(?!\w)','Mademoiselle','abbreviation')
    apply(r'\bMlles\.?(?!\w)','Mesdemoiselles','abbreviation')
    normalized,original_rules=baseline(text)
    return normalized,list(dict.fromkeys(rules+original_rules))
