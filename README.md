<p align="center"><img src="https://raw.githubusercontent.com/IvannP39/Paradis-TTS-release/main/assets/paradis-banner.png" alt="Paradis — Synthèse vocale française sur CPU" width="100%"></p>

<p align="center">
  <a href="https://github.com/IvannP39/Paradis-TTS-release/releases/tag/v0.1.4"><img alt="Release v0.1.4" src="https://img.shields.io/badge/release-v0.1.4-9ee8cb?style=flat-square"></a>
  <a href="https://huggingface.co/IvannP/Paradis-TTS"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging_Face-mod%C3%A8les-ffd681?style=flat-square"></a>
  <a href="https://pypi.org/project/paradis-tts/"><img alt="PyPI" src="https://img.shields.io/pypi/v/paradis-tts?style=flat-square"></a>
  <img alt="CPU et ONNX" src="https://img.shields.io/badge/runtime-CPU_%C2%B7_ONNX-c3dbe6?style=flat-square">
  <a href="https://github.com/IvannP39/Paradis-TTS-release/blob/main/LICENSE"><img alt="Apache 2.0" src="https://img.shields.io/badge/code_%26_poids-Apache_2.0-c3dbe6?style=flat-square"></a>
</p>

**Une voix française compacte, directement dans votre application.**
Paradis-TTS transforme du texte en audio sur CPU, avec 8,08 millions de paramètres,
des poids ONNX FP32 ou INT8, et un post-traitement intégré. Après le premier
téléchargement, la synthèse fonctionne entièrement hors ligne.

> « À ma connaissance, l'un des plus petits modèles open source de synthèse
> vocale en français avec une voix naturelle. » — Ivann

Créé en solo par **Ivann**, avec **0 € de budget d'entraînement**, et entraîné
principalement en local sur un **laptop RTX 3060**. Budget hors matériel et électricité.

[Écouter la voix](https://huggingface.co/IvannP/Paradis-TTS#écouter) ·
[Télécharger v0.1.3](https://github.com/IvannP39/Paradis-TTS-release/releases/tag/v0.1.3) ·
[Documentation Python](https://github.com/IvannP39/Paradis-TTS-release/blob/main/docs/PYTHON_PACKAGE.md)

## Essayer

Python 3.10–3.12 :

```bash
python -m pip install paradis-tts
```

```python
from paradis_tts import ParadisTTS

tts = ParadisTTS()
audio = tts("Bonjour tout le monde.")
tts.save(audio, "bonjour.wav")
```

La construction de `ParadisTTS()` télécharge les poids FP32 depuis Hugging Face
et les garde en cache après vérification SHA256. Aucun GPU, PyTorch ni
Kokoro n'est nécessaire. `audio` est un tableau NumPy mono, `float32`, à 24 kHz.
Le package est disponible sur [PyPI](https://pypi.org/project/paradis-tts/).

La progression du téléchargement et les étapes de chargement s'affichent par
défaut. Pour un chargement silencieux : `ParadisTTS(verbose=False)`.
Le package v0.1.4 réutilise les modèles v0.1.3 et leur cache.

## Choisir sa version

| | FP32 | INT8 |
|---|---:|---:|
| Archive à télécharger | 30.21 Mo | 8.21 Mo |
| Trois graphes ONNX | 32,82 Mo | 9,19 Mo |
| Paramètres | 8 082 881 | 8 082 881 |
| Poids | Flottants | Entiers 8 bits, déquantifiés au calcul |

```python
tts = ParadisTTS(variant="int8")
tts = ParadisTTS(weight_kind="raw")  # poids bruts finaux, au lieu de la moyenne EMA
tts = ParadisTTS(post_traitement=False)  # sortie brute
tts = ParadisTTS(model_folder="chemin/vers/fp32")  # archive locale décompressée
```

Le profil audio par défaut associe le filtre de phase 97, une restauration de
présence jusqu'à 2,5 dB et un adoucissement des aigus à 1,5 dB.
L'adoucissement dynamique est désactivé depuis la 0.1.3.
[Réglages et usage hors ligne](https://github.com/IvannP39/Paradis-TTS-release/blob/main/docs/PYTHON_PACKAGE.md).
L'INT8 réduit le téléchargement ; il ne garantit pas une exécution plus rapide
et peut modifier la voix. Le FP32 est le choix par défaut.

## Vitesse sur CPU — français

40 phrases françaises × 4 passages par moteur, CPU Ryzen 7 5800H, Windows, 2 threads. Poids EMA finaux à 10 000 étapes ; filtres optimisés de la 0.1.3.

| Modèle | Calcul pour 10 s de voix ↓ | Vitesse |
|---|---:|---:|
| Piper français `siwis-medium` | 0.570 s | 17.5× |
| **Paradis FP32*** | 0.695 s | 14.4× |
| **Paradis INT8*** | 0.695 s | 14.4× |
| Kokoro français `ff_siwis` | 5.924 s | 1.7× |

\* Avec un léger post-traitement, inclus dans les temps mesurés.

Appels texte → audio complets, phonémisation et traitements natifs inclus ; chargement et écriture WAV exclus. ONNX Runtime pour Paradis/Piper, PyTorch pour Kokoro. Les durées audio diffèrent entre moteurs ; la vitesse ne classe pas la qualité. Le candidat mesuré et la release utilisent les mêmes calculs audio et poids. [Protocole, mesures par phrase et incertitudes](https://github.com/IvannP39/Paradis-TTS-release/tree/main/evaluation/speed_20261016).

## Qualité : mesures historiques — v0

Ces mesures concernent les anciens poids bruts à 6 500 étapes. Elles n'ont pas
été recalculées pour l'EMA de v0.1.3 ; elles ne mesurent pas son gain de qualité.

300 phrases inédites, articles séparés du corpus de distillation. CPU AMD Ryzen 7 5800H, Windows, deux threads configurés dans ONNX Runtime/PyTorch.

| Moteur | WER ↓ | DNSMOS global ↑ | RTF médian ↓ | Pic RSS |
|---|---:|---:|---:|---:|
| Paradis FP32 | 12.51 % | 3.36 | 0.175 | 374 Mio |
| Paradis INT8 | 12.70 % | 3.35 | 0.183 | 355 Mio |
| Kokoro ff_siwis | 12.33 % | 3.42 | 0.719 | 4405 Mio |
| Piper siwis medium | 16.47 % | 3.25 | 0.119 | 497 Mio |

WER : taux d’erreurs de mots mesuré par Whisper Small en français. DNSMOS P.835 est un proxy parole/bruit, **pas une note d’écoute humaine ni une mesure suffisante de naturalité**. RTF < 1 signifie une génération plus rapide que la lecture. Les mesures ont eu lieu sur un portable partagé avec un entraînement GPU et d’autres tâches CPU ; elles ne constituent pas un classement en conditions isolées. [Protocole, scores par phrase et limites](https://github.com/IvannP39/Paradis-TTS-release/tree/main/evaluation).

## D'où vient la voix ?

Paradis adapte la méthode de [Paradee](https://github.com/sahilmahendrakar/paradee)
au français et distille [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M),
voix `ff_siwis`. La v0.1.3 assemble le modèle texte à 8 000 étapes et le décodeur
A affiné, avec sa **moyenne EMA finale à 10 000 étapes de continuation GAN**
(décroissance 0,999). Les poids bruts finaux restent disponibles en alternative.
Le corpus de distillation comporte
12 000 phrases françaises d'apprentissage, 500 de validation et 500 de test.

**Limites :** une seule voix, français uniquement, quelques artefacts résiduels
sur les consonnes, normalisation limitée aux formes courantes. La qualité et
les prononciations du teacher restent des limites. Les autres langues et le
clonage de voix ne sont pas pris en charge. Le modèle reste expérimental.

## Pour aller plus loin

[Évaluation reproductible](https://github.com/IvannP39/Paradis-TTS-release/blob/main/evaluation/README.md) ·
[Architecture et exports](https://github.com/IvannP39/Paradis-TTS-release/blob/main/docs/MODEL.md) ·
[Normalisation française](https://github.com/IvannP39/Paradis-TTS-release/blob/main/docs/PHASE10_NORMALIZATION.md) ·
[Code du runtime](https://github.com/IvannP39/Paradis-TTS-release/tree/main/paradis_tts) ·
[Licences et attributions](https://github.com/IvannP39/Paradis-TTS-release/blob/main/docs/RELEASE_LICENSES.md)

Code et poids : Apache-2.0. Textes Wikipédia : licences CC BY-SA séparées.
Le G2P installé par pip comprend des composants GPL ; le runtime complet n'est
pas présenté comme entièrement permissif. Voir les notices avant redistribution.
