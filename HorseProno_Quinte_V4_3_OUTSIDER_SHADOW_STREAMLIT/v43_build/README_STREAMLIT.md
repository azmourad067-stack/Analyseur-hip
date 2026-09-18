# HorseProno Quinté V4.2 — Streamlit

## Déploiement

1. Décompresse le ZIP à la racine du dépôt GitHub.
2. Dans Streamlit Community Cloud, choisis le dépôt et la branche.
3. **Main file path : `streamlit_app.py`**.
4. Python 3.12 recommandé.

Le ranker LambdaMART est lu par `portable_lgbm.py`; le package `lightgbm` n'est pas requis.

## Supabase (optionnel)

L'app peut fonctionner avec l'historique CSV embarqué. Pour compléter l'historique au-delà de sa date de fin, configure dans Streamlit Secrets :

```toml
SUPABASE_URL = "..."
SUPABASE_PUBLISHABLE_KEY = "..."
```

Pour les snapshots de marché, une clé serveur peut être ajoutée uniquement dans Streamlit Secrets :

```toml
SUPABASE_SECRET_KEY = "..."
```

Ne jamais committer une clé serveur dans GitHub.

## V4.2 PLAT

La V4.2 conserve la fusion V2/V3. En PLAT, elle adapte uniquement le coefficient marché selon l'écart de poids pré-course. Voir `README_V42.md` pour le protocole et les résultats.
