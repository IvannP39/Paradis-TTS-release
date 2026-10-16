# Paradis-TTS 0.1.4

La bibliothèque indique désormais ce qu'elle fait dès `ParadisTTS()` :
téléchargement Hugging Face avec progression, vérification SHA256, extraction,
utilisation du cache, chargement CPU et confirmation que le modèle est prêt.

Les messages sont activés par défaut et envoyés sur `stderr`. Ils peuvent être
désactivés avec `ParadisTTS(verbose=False)`, `download_model(verbose=False)` ou
`paradis-tts synthesize --quiet`. La synthèse n'ajoute pas de logs par phrase et
le rapport JSON de la CLI reste exploitable sur `stdout`.

Les poids, le post-traitement et le cache des modèles v0.1.3 sont conservés.
Une mise à jour du package ne retélécharge pas un modèle déjà en cache.
Les archives de modèles restent disponibles dans la
[release v0.1.3](https://github.com/IvannP39/Paradis-TTS-release/releases/tag/v0.1.3).

Validation : dix tests d'API, dont la progression au premier téléchargement,
la réutilisation du cache, le mode silencieux avec transfert Hugging Face de
secours et les erreurs de chargement. La publication vérifie aussi un vrai
téléchargement et une synthèse CPU depuis le wheel installé.
