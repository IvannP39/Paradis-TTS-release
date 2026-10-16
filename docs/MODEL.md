# Modèle v0.1.2 et exports

Paradis-TTS est une synthèse française mono-voix distillée depuis Kokoro `ff_siwis`,
selon l'approche de Paradee. La v0.1.2 assemble les poids texte `s+mlp` à 8 000 étapes
et la moyenne EMA finale du décodeur `A_full_phase` à 10 000 étapes de continuation
GAN, avec une décroissance de 0,999. Les poids bruts de cette même étape sont
disponibles avec `weight_kind="raw"`. Les quatre archives (EMA/brut × FP32/INT8)
sont figées et vérifiées par SHA256. La v0 historique reste accessible et utilise
le décodeur brut à 6 500 étapes.

| Composant | Paramètres |
|---|---:|
| Texte, durées et prosodie | 4 231 828 |
| Décodeur et mixeur d'excitation | 3 851 053 |
| Total | 8 082 881 |

Normalisation française → eSpeak NG / phonèmes → `text.onnx` → durées et
caractéristiques → `prosody.onnx` → pitch/énergie → excitation NumPy →
`decoder.onnx` → post-traitement → audio mono 24 kHz.

Les fichiers `text.onnx`, `prosody.onnx` et `decoder.onnx` constituent ensemble
le modèle. Conserver aussi `manifest.json`, `vocab.json` et le profil audio.
Les graphes utilisent des longueurs dynamiques, un batch de 1 et ONNX opset 17.
L'excitation inclut un petit mixeur appris 9→1 exécuté avec NumPy.

FP32 : graphes de 32 819 396 octets. INT8 : 9 186 652 octets, avec 382 tenseurs
de poids réellement stockés en int8, échelles par canal et `DequantizeLinear`.
Le calcul demeure flottant : ce n'est pas une promesse de kernels entiers,
de réduction de RAM ou de gain de vitesse. Les paramètres sont les mêmes.

L'export FP32 est comparé au modèle PyTorch avec exactement la même excitation
sur quatre cas de longueurs et vitesses différentes. Le bruit NumPy n'utilise
pas le même générateur aléatoire que PyTorch ; une même seed entre runtimes ne
signifie donc pas une identité des WAV. Le runtime ONNX est répétable avec une
seed fixe dans les mêmes conditions logicielles.

Les archives et leurs SHA256 sont joints à la release. Les sources du package
CPU sont publiques dans ce dépôt ; elles ne nécessitent aucun accès au dépôt
privé d'expériences d'entraînement. `provenance.json` décrit le checkpoint figé,
les composants et les validations d'export.
