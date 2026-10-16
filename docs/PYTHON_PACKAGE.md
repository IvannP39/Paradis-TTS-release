# Package Python 0.1.4 — modèles v0.1.3

La 0.1.4 affiche par défaut la progression du téléchargement et les étapes de
chargement. Elle réutilise les poids et le cache v0.1.3, sans changement audio.
La 0.1.3 optimise les filtres et désactive l'adoucissement dynamique après
validation à l'écoute le 16 octobre 2026. Les poids neuronaux restent identiques
à la 0.1.2 : moyenne EMA finale à 10 000 étapes. Les anciennes releases sont conservées.

Python 3.10 à 3.12. Runtime CPU : NumPy, ONNX Runtime, Misaki et eSpeak NG.
Ni PyTorch, ni Kokoro, ni GPU ne sont nécessaires à l'inférence.

Le package est publié sur [PyPI](https://pypi.org/project/paradis-tts/).
Le wheel et les sources sont aussi joints à la
[release v0.1.4](https://github.com/IvannP39/Paradis-TTS-release/releases/tag/v0.1.4).

```bash
python -m pip install paradis-tts
```

```python
from paradis_tts import ParadisTTS

tts = ParadisTTS()  # équivalent à post_traitement=True
audio = tts("Bonjour tout le monde.")  # numpy.float32, mono, 24 000 Hz
tts.save(audio, "bonjour.wav")
```

Le premier chargement télécharge le FP32 depuis Hugging Face, à une révision figée, vérifie
son SHA256 puis le conserve dans le cache. Les suivants fonctionnent hors ligne.
La sortie n'est pas jouée automatiquement. Le cache est sous `%LOCALAPPDATA%`
sur Windows, `~/.cache` ailleurs ; `PARADIS_TTS_CACHE` permet de le déplacer.

Depuis 0.1.4, la bibliothèque affiche automatiquement le téléchargement avec
progression, la vérification SHA256, l'extraction, le chargement CPU puis
« Modèle prêt ». Si les fichiers sont déjà en cache, elle le signale sans les
retélécharger. Ces messages vont sur `stderr` ; aucun message n'est ajouté à
chaque synthèse. Pour désactiver ces messages et la barre de progression :

```python
tts = ParadisTTS(verbose=False)
```

`download_model(verbose=False)` permet aussi un téléchargement silencieux.
Cette option ne modifie pas les réglages globaux de Hugging Face et ne masque
pas les exceptions ni les éventuels avertissements des dépendances.

```python
tts = ParadisTTS(variant="int8")
tts = ParadisTTS(weight_kind="raw")  # poids finaux bruts, alternative à l'EMA par défaut
tts = ParadisTTS(local_files_only=True)  # échoue si aucun modèle n'est déjà en cache
tts = ParadisTTS(model_folder="chemin/vers/fp32")  # archive décompressée, aucun téléchargement
tts = ParadisTTS(post_traitement=False)  # désactive tous les filtres
tts.to_file("Le TGV part à 18 h 30.", "train.wav", seed=1729, speed=1.0)
```

`model_folder` doit pointer vers le dossier décompressé contenant `manifest.json`
et les graphes ONNX. Ce chemin est utilisé directement, sans réseau ; un chemin
absent ou invalide provoque une erreur, jamais un téléchargement de remplacement.
`bundle=...` reste un alias compatible avec les anciennes applications.
`variant` sert uniquement au téléchargement automatique : le dossier local
détermine lui-même sa variante et ses poids. Aucun compte Hugging Face n'est nécessaire.
Depuis 0.1.2, les poids par défaut sont la moyenne EMA finale à 10 000 étapes
de continuation. `weight_kind="ema"` la sélectionne aussi explicitement.
Les caches sont séparés par release, type de poids et précision : une mise à
jour vers 0.1.2 ne réutilise pas les anciens poids v0 du cache.

Traitements par défaut depuis 0.1.3 : filtre de phase 97, présence
jusqu'à 2,5 dB, haut du spectre réduit de 1,5 dB. L'adoucissement dynamique
est désactivé à 0 dB, y compris avec les archives antérieures qui indiquent 1 dB.
La réduction des attaques, des syllabes fortes et le lissage sont à zéro.
Le profil local `postprocessing.json` conserve les autres réglages. Passer
explicitement `postprocessing=PostProcessing(...)` (ou charger ce profil avec
`PostProcessing.from_file(...)`) permet de demander un réglage personnalisé,
y compris de réactiver l'adoucissement dynamique.

```python
from paradis_tts import ParadisTTS, PostProcessing

tts = ParadisTTS(postprocessing=PostProcessing(
    dynamic_soften_db=0.0, upper_soften_db=1.5,
    attack_soften_db=0.5, syllable_soften_db=0.0, dynamics_smoothing=0.5,
))
```

Les filtres sont inclus dans l'API et appliqués une seule fois. Pour traiter
séparément un WAV 24 kHz mono PCM16, utiliser `postprocess_wav`. Cela applique
l'adoucissement spectral/dynamique ; le filtre de phase nécessite les signaux
internes et reste propre à la synthèse.

```bash
paradis-tts synthesize --text "Bonjour tout le monde." --output bonjour.wav
paradis-tts synthesize --variant int8 --text "Bonjour." --output bonjour.wav
paradis-tts synthesize --model-folder chemin/vers/fp32 --text "Bonjour." --output bonjour.wav
paradis-tts synthesize --quiet --text "Bonjour." --output bonjour.wav
```

Les logs de la CLI vont sur `stderr` ; son rapport JSON reste seul sur `stdout`.

Construire les distributions : `python -m build`, puis
`python -m twine check dist/*`. Le wheel embarque uniquement le runtime ; les
poids restent dans les assets de release avec SHA256 figés. Le wheel peut être
envoyé à PyPI avec `python -m twine upload dist/*` après authentification du
propriétaire, ou via le workflow Trusted Publishing fourni. Ne pas ajouter de
token dans le dépôt. La version 0.1.1 a été publiée via Trusted Publishing.

Les prochaines releases du package sont publiées automatiquement depuis GitHub,
voir la [configuration Trusted Publishing](PYPI_PUBLISHING.md).

Compatibilité vérifiée localement : Windows, Python 3.11, CPU AMD Ryzen 7 5800H.
Les autres plateformes sont ciblées, sans benchmark local revendiqué.
