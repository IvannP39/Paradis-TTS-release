# Licences de la release v0

Le code Paradis-TTS et les poids distillés publiés dans cette release sont
distribués par le projet sous **Apache-2.0** ; le texte de licence accompagne les
archives. Cette déclaration ne remplace pas les licences des dépendances ou
des textes du corpus.

- Architecture et adaptations d'export : [Paradee](https://github.com/sahilmahendrakar/paradee),
  Sahil Mahendrakar, Apache-2.0, commit `9c8b4d7504cbee7e64de2d0341bb690f0b0ab708`.
- Teacher : [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), hexgrad,
  Apache-2.0, révision `f3ff3571791e39611d31c381e3a41a3af07b4987`, voix `ff_siwis`.
  Aucun poids teacher n'est inclus dans les archives d'inférence.
- La voix teacher déclare une origine [SIWIS](https://doi.org/10.7488/ds/1705),
  CC BY 4.0 : Yamagishi, Junichi ; Honnet, Pierre-Edouard ; Garner, Philip ;
  Lazaridis, Alexandros (2017), *The SIWIS French Speech Synthesis Database, 2016*,
  University of Edinburgh. Aucun enregistrement SIWIS original n'est redistribué.
  Les auteurs et la locutrice d'origine ne sont pas présentés comme associés au projet.
- Corpus texte : Wikipédia française, `20231101.fr`, révision
  `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Originaux CC BY-SA 3.0 ; sélection et
  adaptations CC BY-SA 4.0. Attributions par phrase, URLs et historiques dans
  `provenance/training_text_attribution.jsonl` et `evaluation/heldout_v0.jsonl`
  du dépôt public GitHub.
  Les shards de distillation et l'audio teacher d'apprentissage ne sont pas
  inclus dans les archives. La licence Apache des poids n'efface pas les
  obligations des textes lorsqu'ils sont eux-mêmes redistribués.
- Exemples audio de la release : phrases originales du projet, voix synthétique.
- G2P installé séparément par pip : Misaki (Apache-2.0), phonemizer-fork
  (GPL-3.0), eSpeak NG (GPL-3.0-or-later), distribué via espeakng-loader.
  Les archives ne contiennent aucune bibliothèque native eSpeak. Une application
  qui redistribue ces dépendances doit conserver leurs licences et les sources
  correspondantes selon leurs conditions. Le runtime complet n'est pas présenté
  comme exclusivement permissif.

Évaluation uniquement : Piper (runtime GPL-3.0) et sa voix `fr_FR-siwis-medium`
(provenance SIWIS, CC BY 4.0), Faster Whisper Small (MIT), Microsoft DNSMOS
(dépôt DNS-Challenge, CC BY 4.0). La méthode P.835 est adaptée avec attribution
dans NOTICE, sans revendiquer de validation par Microsoft. Leurs modèles ne sont pas
inclus dans le package ni dans les archives d'inférence.

Le présent document décrit les fichiers effectivement diffusés en v0. Les
révisions et liens des sources sont conservés dans `provenance/sources.json`
du dépôt public. Cette notice n'ajoute aucune garantie sur les droits d'une
voix ou l'absence de mémorisation du teacher.
