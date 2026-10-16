# Nouvelle comparaison CPU française — revue locale

40 phrases françaises identiques au premier essai : 8 courtes, 24 moyennes,
8 longues. Quatre passages par moteur, soit 640 appels chronométrés au total.
Ryzen 7 5800H, Windows, deux threads. Un seul moteur à la fois ; quatre ordres
cycliques équilibrés et ordre des phrases commun à tous les moteurs par tour.

Paradis FP32 et INT8 utilisent le code local figé dans le candidat de release (hashes dans report.json),
les poids EMA finaux à 10 000 étapes des archives publiées v0.1.2, phase 97,
présence 2,5 dB et aigus 1,5 dB. Adoucissement dynamique désactivé (0 dB).
Les hashes du code, des poids et du corpus sont conservés. Le numéro de version du candidat était encore 0.1.2 ; seuls les métadonnées et liens de distribution changent pour la 0.1.3.

Les références sont les mêmes modèles français figés : Piper fr_FR-siwis-medium
sur ONNX Runtime 1.20.1 et Kokoro-82M ff_siwis sur PyTorch 2.6.0 CPU.
Ils sont tous remesurés, sans reprendre les anciens temps. Leurs traitements
natifs et leurs fréquences d'échantillonnage sont conservés.

Chronométrage de l'appel complet texte normalisé → audio, phonémisation et
post-traitements inclus. Chargement, trois échauffements, écritures WAV et
contrôles exclus. Réseau bloqué dans les workers. Charge CPU extérieure et
mémoire disponible consignées. Les durées audio peuvent différer par moteur.
RTF global = somme des temps / somme des durées audio. Incertitude : bootstrap
stratifié par phrase, répétitions regroupées, 3 000 tirages.

Résultats approuvés pour publication par l'auteur. Ce tableau mesure la vitesse, pas la qualité.

Les 640 mesures et les 40 textes attribués sont fournis ici. Les scores de qualité historiques v0 sont séparés de ce benchmark de vitesse. Aucun entraînement ou autre benchmark n'a été lancé en parallèle. Les applications habituelles restent ouvertes ; la charge CPU extérieure maximale estimée vaut 2,09 %.
