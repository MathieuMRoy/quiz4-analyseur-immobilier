# King County Real Estate Analyzer

Application Streamlit pour explorer le marche immobilier du comte de King
et analyser une propriete avec des comparables.

## Fichiers du projet

- `app.py` : application principale
- `kc_house_data.csv` : jeu de donnees
- `dictionnaire_variables.txt` : description des colonnes
- `requirements.txt` : dependances Python
- `.env.example` : exemple de configuration pour Gemini

## Lancement local

1. Creer un environnement virtuel
2. Installer les dependances :

```bash
pip install -r requirements.txt
```

3. Creer un fichier `.env` a partir de `.env.example`
4. Ajouter votre cle Google AI Studio dans `GOOGLE_API_KEY`
5. Lancer l'application :

```bash
streamlit run app.py
```

## Deploiement Streamlit Community Cloud

- Repository : ce dossier GitHub
- Main file path : `app.py`
- Secrets Streamlit :

```toml
GOOGLE_API_KEY="votre_cle_google_ai_studio"
GEMINI_MODEL="gemini-2.5-flash"
```

## Fonctions incluses

- chargement et preparation des donnees avec pandas
- filtres interactifs dans la sidebar
- graphiques matplotlib pour le marche
- analyse de comparables pour une propriete
- deux appels LLM avec Gemini
