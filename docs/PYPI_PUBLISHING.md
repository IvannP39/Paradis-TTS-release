# Publication PyPI

La publication part du dépôt public `IvannP39/Paradis-TTS-release`.
Le dépôt privé de développement ne peut pas lancer le job de publication.

Première publication réussie : **0.1.1**, disponible sur
[PyPI](https://pypi.org/project/paradis-tts/0.1.1/), via le
[workflow GitHub](https://github.com/IvannP39/Paradis-TTS-release/actions/runs/38862835808).
La configuration ci-dessous sert de référence pour maintenir cette liaison.


Dernière publication vérifiée : **0.1.4**, via
[Trusted Publishing](https://github.com/IvannP39/Paradis-TTS-release/actions/runs/38882683855).
Le wheel et les sources téléchargés depuis PyPI ont été vérifiés par SHA256
et joints à la release GitHub et au dépôt Hugging Face. Le workflow teste aussi
un vrai téléchargement des poids et une synthèse CPU avant l'envoi.
La 0.1.4 ajoute les logs de chargement et la progression du téléchargement ;
elle conserve les modèles et le cache v0.1.3.

## Première publication : configuration du compte

Connecté à ton compte PyPI, ouvrir
[Publishing](https://pypi.org/manage/account/publishing/), puis ajouter un
**pending publisher GitHub** pour créer le projet lors du premier envoi :

| Champ | Valeur |
|---|---|
| PyPI project name | `paradis-tts` |
| Owner | `IvannP39` |
| Repository name | `Paradis-TTS-release` |
| Workflow name | `publish-pypi.yml` |
| Environment name | `pypi` |

Utiliser seulement le nom du fichier du workflow, sans `.github/workflows/`.
Si le projet existe déjà sur ton compte, ajouter les mêmes champs dans les
Trusted Publishers du projet. Aucun mot de passe ou token PyPI ne va dans GitHub.
Le compte PyPI doit avoir son adresse e-mail vérifiée et la 2FA configurée.

Ensuite, sur GitHub : **Actions → Publish package to PyPI → Run workflow → main**.
Ce lancement manuel permet la première publication sans recréer la release
historique des modèles `v0`.

## Publications suivantes

1. Incrémenter la version dans `pyproject.toml`, `paradis_tts/api.py` et
   `paradis_tts/release_assets.json`, puis pousser sur `main`.
2. Créer une release GitHub dont le tag correspond exactement à la version du
   package, par exemple `v0.1.2`. La publier déclenche le workflow automatiquement.
3. Le workflow construit le wheel et les sources, vérifie les métadonnées,
   installe le wheel et teste l'API avant d'envoyer les deux fichiers à PyPI.

Le job de publication utilise l'environnement GitHub `pypi` et une identité OIDC
temporaire (`id-token: write`). PyPI refuse de remplacer une version existante :
il faut incrémenter la version pour toute nouvelle distribution.

Les versions du package et des modèles sont indépendantes. Le package 0.1.1
charge toujours les modèles `v0`, épinglés à une révision Hugging Face et à des
SHA256. Une correction du chargeur ne change pas les poids déjà publiés.

[Documentation officielle PyPI](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).
