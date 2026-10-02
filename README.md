# IA de Détection des Maladies du Manioc

Projet de mémoire Master 1 Big Data & IA — Intelligence Artificielle multimodale pour détecter les maladies du manioc et prédire les rendements.

## Résultats

| Modèle | Accuracy |
|---|---|
| Unimodal (CNN ResNet50, images seules) | 83.4% |
| **Multimodal (images + environnement)** | **91.4%** |

L'approche multimodale améliore la précision de **+8 points** par rapport au modèle unimodal.

## Architecture

- **Branche Image** : ResNet50 (transfer learning) → 256 features
- **Branche Environnement** : MLP (température, précipitations, humidité, vent, évapotranspiration) → 32 features
- **Fusion** : Concaténation (288 features) → 128 → 5 classes

## Classes de maladies

| Classe | Maladie |
|---|---|
| CBB | Bactériose |
| CBSD | Striure brune |
| CGM | Marbrure verte |
| CMD | Mosaïque |
| Healthy | Sain |

## Données

- **Images** : [Cassava Leaf Disease Dataset](https://www.kaggle.com/c/cassava-leaf-disease-classification) (21 397 images)
- **Météo** : API Open-Meteo (8 villes de Côte d'Ivoire)
- **Sol** : API SoilGrids/ISRIC (Côte d'Ivoire)
- **Satellites** : API NASA POWER (Côte d'Ivoire)

## Démo en ligne

L'application Gradio est disponible sur [Hugging Face Spaces](https://huggingface.co/spaces/Nabi07madawa/manioc-disease-detection).

## Utilisation locale

```bash
pip install -r requirements.txt
python app.py
```

## Technologies

Python, PyTorch, ResNet50, Gradio, Google Colab (GPU T4), Anaconda
