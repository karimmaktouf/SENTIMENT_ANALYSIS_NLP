# 🎭 Twitter Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Système automatisé d'analyse de sentiment sur Twitter utilisant le Machine Learning et l'explicabilité SHAP

## 📋 Table des matières

- [À propos](#à-propos)
- [Fonctionnalités](#fonctionnalités)
- [Dataset](#dataset)
- [Méthodologie](#méthodologie)
- [Résultats](#résultats)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Technologies](#technologies)
- [Auteur](#auteur)

## 🎯 À propos

Ce projet académique développe un système de classification automatique de sentiments (Positif/Négatif/Neutre) sur des tweets en utilisant des techniques de NLP (Natural Language Processing) et de Machine Learning.

**Objectifs :**
- Développer un pipeline complet de traitement de texte
- Comparer plusieurs algorithmes de classification
- Implémenter l'explicabilité via SHAP
- Atteindre une performance >85% d'accuracy

## ✨ Fonctionnalités

- 🧹 **Preprocessing avancé** : Nettoyage, tokenization, lemmatization, stemmatization
- 🏷️ **Labelling automatique** : Utilisation de VADER pour la classification
- 🔢 **Vectorisation TF-IDF** : 5000 features avec N-grams (1,2)
- 🤖 **4 modèles comparés** : Passive Aggressive, Random Forest, XGBoost, Naive Bayes
- 🔍 **Explicabilité SHAP** : Analyse de l'importance des features
- 💾 **Modèle déployable** : Sauvegarde en .pkl pour production

## 📊 Dataset

- **Source** : Twitter API
- **Taille** : 17,744 tweets (anglais uniquement)
- **Période** : Mars 2023
- **Distribution** : 
  - Négatif : 46.5%
  - Positif : 42.8%
  - Neutre : 10.7%

## 🔬 Méthodologie

### Pipeline complet :

1. **Collecte et Nettoyage** : Filtrage des tweets en anglais (88.7% retenus)
2. **Preprocessing du Texte** : 
   - Suppression URLs, hashtags, mentions
   - Expansion contractions (can't → cannot)
   - Tokenization, Lemmatization, Stemmatization
   - Suppression stopwords
3. **Labelling VADER** : Classification automatique en 3 classes
4. **Vectorisation TF-IDF** : 5000 features, N-grams (1,2)
5. **Entraînement** : 4 modèles testés avec split 80/20
6. **Évaluation** : Comparaison via Accuracy, F1-Score, Precision, Recall
7. **Explicabilité SHAP** : Analyse des features importantes
8. **Déploiement** : Sauvegarde modèle + vectorizer

## 🏆 Résultats

### Comparaison des modèles :

| Modèle | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| **Passive Aggressive** | **86.33%** | **86.25%** | **86.25%** | **86.33%** |
| Random Forest | 85.01% | 84.91% | 84.89% | 85.01% |
| XGBoost | 84.87% | 84.56% | 84.73% | 84.87% |
| Naive Bayes | 79.77% | 78.24% | 80.81% | 79.77% |

### Insights SHAP :

**Mots-clés négatifs :**
- suicide, negativity, lonely, hopeless, grief, bad

**Mots-clés positifs :**
- love, heart, feeling, like, romantic, good

→ Le modèle a correctement appris les associations émotionnelles ✅

## 🚀 Installation

### Prérequis :
- Python 3.8+
- pip

### Étapes :

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
```

2. **Créer un environnement virtuel (recommandé)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Télécharger les ressources NLTK**
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 💻 Utilisation

### 1. Entraîner le modèle

Ouvrez et exécutez le notebook Jupyter :
```bash
jupyter notebook projectt.ipynb
```

### 2. Utiliser le modèle pré-entraîné

```python
import pickle
import pandas as pd

# Charger le modèle
model = pickle.load(open('models/pac.pkl', 'rb'))
vectorizer = pickle.load(open('models/tfidf_v.pkl', 'rb'))
encoder = pickle.load(open('models/label_encoder.pkl', 'rb'))

# Prédire sur un nouveau tweet
tweet = ["I love this amazing product!"]
tweet_vectorized = vectorizer.transform(tweet)
prediction = model.predict(tweet_vectorized)
sentiment = encoder.inverse_transform(prediction)[0]

print(f"Sentiment: {sentiment}")  # Output: Positive
```

## 🛠️ Technologies

- **Python 3.8+**
- **Pandas** - Manipulation de données
- **NumPy** - Calculs numériques
- **Scikit-learn** - Machine Learning
- **NLTK** - Traitement du langage naturel
- **VADER** - Analyse de sentiment
- **XGBoost** - Algorithme de boosting
- **SHAP** - Explicabilité du modèle
- **Matplotlib/Seaborn** - Visualisation

## 📈 Améliorations futures

- [ ] Implémenter BERT/RoBERTa pour +10-15% d'accuracy
- [ ] Détection de sarcasme et ironie
- [ ] Analyse multilingue
- [ ] Interface web (Streamlit/Gradio)
- [ ] Analyse en temps réel (streaming)
- [ ] Rééquilibrage des classes (SMOTE)

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 👤 Auteur

**[Votre Nom]**
- GitHub: [@votre-username](https://github.com/votre-username)
- LinkedIn: [Votre Profil](https://linkedin.com/in/votre-profil)
- Email: votre.email@example.com

## 🙏 Remerciements

- Dataset fourni par [Twitter API]
- VADER Sentiment Analysis
- Communauté Scikit-learn et SHAP

---

⭐ Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !
