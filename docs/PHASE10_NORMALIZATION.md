# Phase 10 — normalisation française d’inférence

Le runtime ONNX utilise `fr-runtime-2`, dans
`paradis_tts/normalization_v2.py`. Le normaliseur `fr-baseline-1` employé
pour le corpus et l’entraînement reste intact. La nouvelle couche appelle
les règles historiques après avoir développé les formes d’inférence ajoutées.
Les exports et les sidecars WAV enregistrent explicitement cette version.

| Entrée | Lecture prévue |
| --- | --- |
| `12,50 €` | douze euros et cinquante centimes |
| `18 h 30` | dix-huit heures trente |
| `2026` | deux mille vingt-six |
| `25 %` | vingt-cinq pour cent |
| `3,14` | trois virgule un quatre |
| `3 km`, `2 kg` | trois kilomètres, deux kilogrammes |
| `M. Dupont`, `Mme Martin`, `Dr. Durand` | Monsieur Dupont, Madame Martin, Docteur Durand |
| `SNCF`, `TGV` | ès ène cé èf, té gé vé |
| `https://exemple.fr` | exemple point èf ère |
| `contact@exemple.fr` | contact arobase exemple point èf ère |

Les dates et heures impossibles reconnues par les règles sont refusées.
Les décimales gardent chaque chiffre après la virgule, comme pendant le
training. Les apostrophes, espaces, accents et marques nasales IPA restent
traités par le frontend historique. Aucun phonème inconnu n’est ignoré et
les textes longs sont segmentés sans troncature silencieuse.

La liste de sigles est volontairement bornée : SNCF, TGV, RATP, EDF, CNRS,
USB, UE, IA, TV et GPS. Les titres Mlle/Mlles sont également développés.
Les adresses prises en charge sont des URL et e-mails simples : domaine,
suffixe, chemin simple, tirets et chiffres. Les URL complexes avec paramètres,
ports, identifiants et encodages ne bénéficient pas d’une garantie de lecture
complète. Il n’y a pas de désambiguïsation universelle des sigles, noms propres,
formules ou nombres de téléphone.

```python
from paradis_tts.normalization_v2 import normalize_sentence
text, rules = normalize_sentence("Le TGV part à 18 h 30.")
```

L’API `OnnxTTS` appelle automatiquement cette normalisation. Les anciennes
écoutes et l’API de référence Torch continuent à employer la version du
training. Le corpus et ses fingerprints ne sont pas réécrits pendant le run.

Validation : tests ciblés chiffres/prix/date/heure/unités/titres, sigles,
URL/e-mail, caractères accentués, entrées invalides ; génération effective
de WAV dans l’environnement portable sans Torch ni accès réseau. Les preuves
sont conservées dans [evaluation/normalization_validation.json](../evaluation/normalization_validation.json).
