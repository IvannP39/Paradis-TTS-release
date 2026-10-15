# Évaluation v0

300 phrases françaises de 300 articles Wikipédia, sans article commun avec les
12 000 phrases d'apprentissage, les 500 de validation et les 500 de test.
La sélection déterministe exclut les doublons et quasi-doublons par Jaccard
sur trigrammes de mots (seuil 0,85). Répartition : 60 courtes, 180 moyennes,
60 longues. Le jeu est figé avant les mesures ; aucun réglage ne doit être
choisi à partir de ses scores.

Les phrases proviennent de `wikimedia/wikipedia`, snapshot français `20231101.fr`,
révision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Les originaux restent
CC BY-SA 3.0 ; les adaptations/sélections sont CC BY-SA 4.0. Chaque ligne de
`heldout_v0.jsonl` attribue les contributeurs, lie l'article et son historique,
et conserve les offsets et la révision du snapshot. Ce fichier n'est pas
couvert par la licence Apache du code.

Comparaison : Paradis FP32 et INT8 avec les mêmes traitements, Kokoro `ff_siwis`
et Piper `fr_FR-siwis-medium`. Chaque moteur reçoit le même texte normalisé.
Le RTF inclut G2P, génération et traitements du moteur, exclut chargement,
échauffement, ASR et écriture des WAV. Tous les moteurs tournent sur CPU,
deux threads configurés dans ONNX Runtime/PyTorch, dans des processus distincts.
ORT utilise un thread inter-op ; les bibliothèques numériques auxiliaires
conservent leurs réglages par défaut. Le pic RSS est échantillonné toutes
les 20 ms et inclut le processus de synthèse complet. Piper conserve sa
normalisation de volume standard ; aucun traitement Paradis ne lui est ajouté.

WER : Faster Whisper Small FP16 sur GPU pour la mesure publiée, français, beam 5, température 0,
sans VAD ni conditionnement sur les phrases précédentes. La même normalisation
est appliquée aux références et transcriptions (minuscules, chiffres développés,
ponctuation supprimée, apostrophes et tirets séparateurs, accents conservés).
Le WER global est calculé sur les sommes de substitutions, suppressions et
insertions, divisées par le nombre total de mots de référence.
Le rapport donne aussi un intervalle à 95 % par bootstrap de 2 000 rééchantillonnages
de phrases, seed 1729. Il ne couvre pas les biais propres à l'ASR ou au corpus.

Qualité : DNSMOS P.835 non personnalisé de Microsoft, signal/bruit/global.
Fenêtres de 9,01 s, pas de 1 s, répétition des clips courts et calibration
polynomiale officielle. Rééchantillonnage à 16 kHz par filtre polyphase SciPy.
P.808 n'est pas calculé. **Ces scores sont un proxy de parole/bruit entraîné
pour le débruitage, pas un MOS humain ni une mesure suffisante de naturalité TTS.**

Limites : même domaine Wikipédia que l'entraînement ; possible recouvrement
avec les corpus originaux de Kokoro/Piper ; noms propres difficiles pour l'ASR.
Le portable exécute un entraînement GPU indépendant pendant les mesures CPU :
les temps décrivent ces conditions et ne justifient aucun classement général.
L'INT8 désigne le stockage des poids ; les calculs restent flottants.

Reproduction : `scripts/evaluate_release.py`, `config_v0.json`, puis les
étapes `synthesize` (une par moteur), `score` et `aggregate`. Les WAV et logs
intermédiaires restent locaux ; les scores par phrase et résultats agrégés
sont publiés. Les moteurs d'évaluation ne sont pas des dépendances du package.

Les textes et transcriptions liés au jeu gardent les attributions et licences
CC BY-SA décrites ci-dessus. Les mesures numériques ne sont pas des avis humains.

Exemple de reproduction (dans des environnements d'évaluation séparés du package) :

```bash
python -m pip install -r requirements-evaluation.txt
# Kokoro nécessite en plus son runtime et PyTorch CPU, dans son environnement dédié.
python scripts/evaluate_release.py synthesize --variant fp32 --bundle chemin/vers/exports
python scripts/evaluate_release.py synthesize --variant int8 --bundle chemin/vers/exports
python scripts/evaluate_release.py synthesize --variant piper
python scripts/evaluate_release.py synthesize --variant kokoro
python scripts/evaluate_release.py score --asr-device cuda --asr-compute float16
python scripts/evaluate_release.py aggregate
python scripts/audit_evaluation_split.py
```

`chemin/vers/exports` contient les dossiers `fp32` et `int8` décompressés.
Pour une mesure ASR CPU, utiliser `--asr-device cpu --asr-compute int8` ; les
chiffres peuvent différer légèrement de l'ASR GPU de la publication. Le GPU
d'évaluation nécessite CUDA/cuDNN compatibles avec CTranslate2. Les révisions
des modèles sont figées dans `config_v0.json` et les versions installées dans
le lock Windows. Le runtime de synthèse public reste exclusivement CPU.
