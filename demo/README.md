# Le petit studio Paradis

Une démo locale pour écrire une phrase en français, écouter la voix et
télécharger son WAV. Le temps de génération, la durée audio et la vitesse par
rapport au temps réel s'affichent après chaque génération.

## Lancer

Installer **Python 3.10, 3.11 ou 3.12**, télécharger ce dossier, puis ouvrir un
terminal dans ce dossier et lancer :

```bash
python lancer.py
```

Le lanceur crée un environnement isolé dans `demo/.venv` et installe le package
public `paradis-tts`. Le navigateur s'ouvre sur **http://127.0.0.1:7860**. Au
premier lancement, le modèle FP32 est téléchargé automatiquement depuis Hugging
Face, avec sa progression dans le terminal. Attendre l'indicateur « Prêt à
parler », saisir une phrase et cliquer sur **Générer la voix**.

Les lancements suivants réutilisent l'environnement et le cache du modèle.
Ils fonctionnent hors ligne. Aucun compte, GPU, PyTorch ou Node.js n'est requis.
Le post-traitement du package est activé par défaut. Arrêt : **Ctrl+C** dans le
terminal. La démo accepte jusqu'à 1 000 caractères par génération.

## Si le package est déjà installé

Avec le Python de cet environnement :

```bash
python app.py
```

Pour choisir un autre port, ne pas ouvrir automatiquement le navigateur, ou
charger une archive ONNX déjà décompressée :

```bash
python lancer.py --port 7861 --no-browser
python lancer.py --model-folder "chemin/vers/fp32"
```

Depuis le dossier du projet de développement, par exemple :

```powershell
.venv-runtime/Scripts/python.exe demo/app.py --model-folder models/final/fp32
```

## Ce que mesure le compteur

Le temps est mesuré dans le backend autour de `tts(texte)`, avec
`time.perf_counter()`. Il comprend la normalisation, la phonémisation, le
calcul CPU et les filtres. Il exclut le chargement/téléchargement initial du
modèle, l'encodage WAV et le transfert vers le navigateur. Le premier appel
peut être plus lent que les suivants. « 10× » signifie dix fois plus rapide
que la lecture en temps réel ; les performances dépendent de l'ordinateur.

## Fichiers

- `lancer.py` : installation isolée au premier lancement, puis démarrage.
- `app.py` : petit serveur HTTP Python, modèle chargé une seule fois.
- `index.html` : interface autonome HTML/CSS/JavaScript, sans CDN ni compilation.
- `requirements.txt` : une seule dépendance directe, le package publié.

Les requêtes restent sur l'ordinateur. Internet sert à l'installation et au
premier téléchargement ; les boutons GitHub/Hugging Face ouvrent ces sites.
Les audios sont créés en mémoire ; aucun historique n'est enregistré sur disque.
Le bouton de téléchargement permet de conserver un WAV mono PCM16 à 24 kHz.
Le serveur écoute uniquement sur `127.0.0.1` et exécute une génération à la fois.
Il est destiné au test local, pas à un hébergement public.
