# RNN POS Tagger – Training & Prediction

Ce dépôt contient deux scripts principaux permettant :
1. **l’entraînement** d’un modèle RNN pour le POS tagging (UPOS)
2. **la prédiction / annotation** de fichiers CoNLL-U à l’aide d’un modèle entraîné

Le format de données utilisé est **CoNLL-U**.

---

## 📁 Structure minimale du projet

```
.
├── train_postag.py          # script d'entraînement
├── predict_postag.py        # script de prédiction
├── model_postag.py          # définition du modèle RNN
├── use_conllulib.py         # lecture CoNLL-U + utilitaires
├── models/                  # modèles sauvegardés
│   └── model_e20_l40_bs32.pt
├── pred/                    # dossiers des prédictions
└── README.md
```

---

## 🧱 Prérequis

### Python
- Python **≥ 3.8**

### Librairies Python
```bash
pip install torch tqdm conllu
```

---

## 📚 Données

Les fichiers doivent être au format **CoNLL-U**, par exemple :

- `sequoia-ud.parseme.frsemcor.simple.train`
- `sequoia-ud.parseme.frsemcor.simple.dev`
- `sequoia-ud.parseme.frsemcor.simple.test`

---

## 🚀 Entraînement du modèle

### Commande minimale

```bash
python train_postag.py -t path/to/train.conllu -d path/to/dev.conllu
```

### Options disponibles

| Option | Description | Défaut |
|------|------------|--------|
| `-t`, `--train_file` | Fichier d’entraînement CoNLL-U | requis |
| `-d`, `--dev_file` | Fichier de développement CoNLL-U | requis |
| `-e`, `--epochs` | Nombre d’époques | 15 |
| `-m`, `--max_len` | Longueur max des séquences | 40 |
| `-b`, `--batch_size` | Taille des batchs | 32 |

### Exemple complet

```bash
python train_postag.py   -t ../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train   -d ../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev   -e 20   -m 40   -b 32
```

### Sortie

À la fin de l’entraînement, le modèle est sauvegardé automatiquement dans le dossier `models/` :

```
models/model_e20_l40_bs32.pt
```

Chaque fichier contient :
- le vocabulaire des mots
- le vocabulaire des tags
- les poids du modèle
- les hyperparamètres

Ce format permet une **traçabilité complète** des expériences.

---

## 🔁 Entraîner plusieurs modèles

Un script Bash permet de lancer plusieurs entraînements avec différentes configurations :

```bash
./run_train_grid.sh
```

Cela permet de comparer plusieurs valeurs de :
- epochs
- max_len
- batch_size

---

## 🔮 Prédiction / Annotation

### Commande minimale

```bash
python predict_postag.py -i path/to/input.conllu -m models/model_e20_l40_bs32.pt
```

### Options disponibles

| Option | Description | Défaut |
|------|------------|--------|
| `-i`, `--input_file` | Fichier CoNLL-U à annoter | requis |
| `-m`, `--model_file` | Modèle entraîné (.pt) | requis |
| `-o`, `--output_file` | Préfixe du fichier de sortie | pred |
| `-l`, `--max_len` | Longueur max des séquences | 40 |
| `-b`, `--batch_size` | Batch size | 32 |
| `-s`, `--seed` | Graine aléatoire | 37 |

---

## 🏷️ Format du fichier de sortie

Le nom du fichier de prédiction est généré automatiquement selon le format :

```
<prefix>-e<EPOCHS>-l<MAX_LEN>-i-s<SEED>-<split>.pred
```

### Exemple

Commande :
```bash
python predict_postag.py   -i sequoia-ud.parseme.frsemcor.simple.dev   -m models/model_e20_l32_bs32.pt   -o pred/sequoia   -l 32   -s 37
```

Fichier généré :
```
pred/sequoia-e20-l32-i-s37-dev.pred
```

---

## 📊 Évaluation

Les fichiers `.pred` peuvent être évalués avec un script externe afin d’obtenir :
- Accuracy globale UPOS
- Accuracy sur les mots OOV

Exemple de résultats :

```
Accuracy on all upos: 92.74 (8998 / 9702)
Accuracy on OOV upos: 59.05 (522 / 884)
```

