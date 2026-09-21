# HorseProno M8.1 — Top 3 Place R1

Application Streamlit séparée pour construire un Top 3 de chevaux susceptibles de finir dans les trois premiers des courses R1.

## Emplacement recommandé dans le dépôt

- `horse_prono_v2/streamlit_m8_1_place.py`
- `horse_prono_v2/scripts/m8_1_top3_place_r1.py`

L'application réutilise `app_core` et le modèle M8 #8 déjà présents dans HorseProno.

## Streamlit Community Cloud

Main file path :

`horse_prono_v2/streamlit_m8_1_place.py`

Secrets nécessaires :

```toml
SUPABASE_URL = "..."
SUPABASE_KEY = "..."
```

## Utilisation

1. Choisir une date.
2. L'application charge automatiquement le programme PMU.
3. Seules les courses R1 sont affichées.
4. Choisir une course.
5. Cliquer sur `Construire les 3 chevaux placés`.
6. Le Top 3 M8.1 et le classement complet sont affichés.

Aucun CSV d'entrée n'est nécessaire.
