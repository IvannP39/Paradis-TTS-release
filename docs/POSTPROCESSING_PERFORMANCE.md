# Optimisation locale du post-traitement

Les mesures d'optimisation ci-dessous utilisent les réglages de la 0.1.2 : EMA finale à
10 000 étapes, phase 97, présence 2,5 dB, adoucissements 1 dB et 1,5 dB.
La release 0.1.3 reprend ces optimisations, puis désactive l'adoucissement
dynamique après la comparaison d'écoute décrite ci-dessous.

## Changements

- Réutiliser la STFT du signal brut entre correction de phase et restauration
  de présence.
- Calculer le lissage de phase uniquement dans la bande corrigée, en conservant
  tous les voisins temporels nécessaires au filtre. Modifier uniquement les
  cases fréquence/temps effectivement corrigées.
- Reconstruire le signal par blocs NumPy plutôt que par boucle Python sur toutes
  les trames. Réutiliser le motif de normalisation du recouvrement, en préservant
  les bords et l'ordre d'addition des contributions.
- Mettre en cache les fenêtres et courbes de gain immuables. Les caches de grandes
  courbes ont quatre entrées au maximum et excluent les FFT supérieures à 2²⁰.
- Éviter le calcul des gains dynamiques sur les fréquences dont le gain vaut
  exactement un ; conserver la précision de calcul précédente.

Pour ces mesures, aucun filtre désactivé, aucun coefficient modifié, aucune nouvelle dépendance.
Les fréquences non corrigées évitent un aller-retour angle/exponentielle : cela
produit des différences au niveau des arrondis float32, pas une identité binaire.

## Mesures

Le 16 octobre 2026, après ces mesures puis une comparaison d'écoute sur trois
phrases, l'utilisateur valide la suppression de l'adoucissement dynamique.
Le runtime 0.1.3 passe donc à **0 dB**, y compris lorsqu'un ancien profil de
modèle indique 1 dB. Phase 97, présence et haut du spectre à 1,5 dB sont conservés.
Le filtre reste accessible par réglage explicite pour reproduire les anciennes
écoutes. Les archives historiques sont conservées ; la 0.1.3 fournit de nouvelles
archives avec le code et le profil mis à jour, sans modifier les poids neuronaux.

Le profilage séparé sur 40 phrases × six passages attribuait à cet
adoucissement 73 ms pour 10 s d'audio, soit 35,8 % du temps du post-traitement
optimisé. Cela ne constitue pas une mesure du gain total de synthèse après
désactivation. Les chiffres suivants décrivent l'optimisation précédente,
avec le filtre encore actif.

CPU Ryzen 7 5800H, Windows, deux threads. Mêmes 40 phrases françaises que dans
la comparaison locale de vitesse ; mêmes poids, graines et sessions ONNX.

| Mesure | Résultat |
|---|---:|
| Temps des filtres seuls | −29,5 % ; accélération 1,419× |
| Temps de synthèse complète | −9,9 % ; accélération 1,109× |
| Calcul pour 10 s de voix, avant → après | 0,838 s → 0,755 s |
| IC 95 % de l'accélération complète | 1,105–1,114× |
| Écart absolu maximal observé | 1,1921 × 10⁻⁷ |
| Erreur RMS relative maximale observée | 6,36 × 10⁻⁸ |

Filtres seuls : formes d'onde neuronales figées, six passages de 40 phrases par
version. Synthèse complète : quatre passages de 40 phrases par version, ancien
et optimisé alternés par phrase (320 appels). Chargement et écritures WAV exclus.
Intervalle par bootstrap stratifié des phrases, 3 000 tirages avec répétitions
regroupées. Ces mesures quantifient l'optimisation sur ce CPU ; les autres moteurs
n'ont pas été remesurés dans ce protocole.

Validation : neuf tests DSP, cinq tests API, comparaison des sorties sur les
40 phrases, parité avec les traitements PyTorch sur trois vraies synthèses,
et textes découpés en plusieurs segments en FP32 et INT8.

Les résultats, références figées et six WAV d'écoute restent dans
`artifacts/private/filter_optimization_20261016/`. `comparison.html` permet
l'écoute avant/après ; `report.json` contient les mesures et les empreintes du code.
