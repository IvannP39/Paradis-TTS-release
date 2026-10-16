# Paradis-TTS 0.1.3

La 0.1.3 réduit le coût du post-traitement CPU. L'adoucissement dynamique à
1 dB est désactivé après comparaison d'écoute et validation par l'auteur.
Phase 97, présence jusqu'à 2,5 dB et adoucissement des aigus à 1,5 dB restent
actifs par défaut, avec compensation de niveau. Attaques, syllabes et lissage
restent à zéro. Le filtre dynamique reste accessible par réglage explicite.

Les optimisations réutilisent les spectres, les fenêtres et les courbes de
gain, limitent les calculs aux bandes utiles et reconstruisent le signal
par blocs NumPy. Le calcul audio ne dépend que de NumPy et ONNX Runtime.
Le client officiel Hugging Face est ajouté pour récupérer une archive si le
téléchargement HTTP direct arrive incomplet ; les SHA256 restent vérifiés.

Les poids FP32 et INT8, EMA et bruts, sont identiques à la 0.1.2 : modèle texte
à 8 000 étapes et décodeur final à 10 000 étapes de continuation GAN.
Les archives sont renouvelées pour inclure le code, les réglages et les WAV
de démonstration à jour ; les anciennes releases et leurs hashes sont conservés.

```bash
python -m pip install --upgrade paradis-tts
```

```python
from paradis_tts import ParadisTTS

tts = ParadisTTS()  # EMA FP32, post-traitement actif sans adoucissement dynamique
audio = tts("Bonjour tout le monde.")
tts.save(audio, "bonjour.wav")
```

Le premier appel du constructeur télécharge les poids depuis une révision
Hugging Face figée et vérifie les SHA256. `model_folder=...` charge un dossier
local sans téléchargement. Les anciens dossiers de modèles utilisent aussi
le nouveau défaut à 0 dB avec ce runtime ; un objet `PostProcessing` explicite
conserve tous ses réglages.

La comparaison CPU française porte sur 40 phrases × quatre passages par moteur,
avec deux threads sur un Ryzen 7 5800H. Générer 10 secondes de voix demande
0,695 s pour Paradis FP32/INT8, 0,570 s pour Piper fr_FR-siwis-medium et
5,924 s pour Kokoro ff_siwis. Phonémisation et post-traitements inclus ;
chargement exclu. Voir les mesures et limites dans `evaluation/speed_20261016/`.
Les résultats ne constituent pas un classement de qualité. Le benchmark utilise
le candidat de la release ; seuls la version et les liens de distribution ont
changé depuis la mesure, pas les calculs audio ni les poids.

Les scores WER/DNSMOS historiques de v0 ne sont pas recalculés et ne doivent
pas être attribués aux poids EMA actuels. La naturalité reste à juger à l'écoute.
