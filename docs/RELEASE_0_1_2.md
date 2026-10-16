# Paradis-TTS 0.1.2

Cette version utilise par défaut la **moyenne EMA finale du décodeur A affiné**,
à 10 000 étapes de continuation GAN. La décroissance EMA vaut 0,999. Le modèle
texte `s+mlp` à 8 000 étapes, la normalisation et le post-traitement restent identiques.

```bash
python -m pip install --upgrade paradis-tts
```

```python
from paradis_tts import ParadisTTS

tts = ParadisTTS()                       # EMA finale, FP32, post-traitement actif
tts = ParadisTTS(variant="int8")          # EMA finale, poids INT8
tts = ParadisTTS(weight_kind="raw")       # poids bruts finaux, FP32
tts = ParadisTTS(weight_kind="raw", variant="int8")
audio = tts("Bonjour tout le monde.")
```

Les quatre archives sont disponibles dans la release GitHub et sur Hugging Face.
Le package télécharge automatiquement les poids choisis depuis une révision
Hugging Face épinglée, puis vérifie les SHA256. Les caches sont séparés par
release, type de poids et précision : les anciens modèles restent disponibles.
`model_folder=...` charge toujours directement un dossier décompressé, sans réseau.

Le mot « brut » dans `weight_kind="raw"` désigne les poids sans moyenne EMA.
Il ne désactive pas le post-traitement ; utiliser `post_traitement=False` pour cela.

Le profil par défaut reste : filtre de phase 97, présence jusqu'à 2,5 dB,
adoucissement dynamique 1 dB et haut du spectre 1,5 dB. Les réglages d'attaque,
de syllabes fortes et de lissage restent désactivés.

Les exports FP32 sont comparés à PyTorch sur quatre phrases et vitesses ; le
post-traitement et l'excitation NumPy passent leurs contrôles de parité. Les
quatre archives sont testées hors ligne sans Torch ni Kokoro, avec génération
WAV, graines fixes, changement de vitesse et découpage des textes longs.
Les poids INT8 sont déquantifiés au calcul ; ils ne garantissent pas un gain de vitesse.

La validation spectrale finale vaut 1,509507 pour l'EMA et 1,509864 pour les poids
bruts. Ces valeurs ne constituent pas une mesure de naturalité. Les scores WER
et DNSMOS publiés pour la v0 historique concernent le décodeur brut à 6 500 étapes ;
ils n'ont pas été recalculés pour cette version. Aucun gain perceptuel chiffré
n'est revendiqué.

[Modèles Hugging Face](https://huggingface.co/IvannP/Paradis-TTS/tree/v0.1.2) ·
[Release GitHub](https://github.com/IvannP39/Paradis-TTS-release/releases/tag/v0.1.2) ·
[Package PyPI](https://pypi.org/project/paradis-tts/0.1.2/)
