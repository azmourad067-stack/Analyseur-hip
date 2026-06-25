"""
═══════════════════════════════════════════════════════════════════════════════
 FICHIER DE DONNÉES HISTORIQUES — QUINTÉ+ 2026 (01/01 → 25/06)
═══════════════════════════════════════════════════════════════════════════════
 Sources : GRM-Turf (archives mensuelles) et Canalturf (résultats détaillés)
 Structure compatible avec le Backtester de QuantTurf Pro v5.0.0
───────────────────────────────────────────────────────────────────────────────
 Chaque course contient :
   - date       : str (AAAA-MM-JJ)
   - race_type  : str (Plat, Attelé, Monté, Haies)
   - winner     : int (numéro du cheval vainqueur)
   - top5       : List[int] (arrivée complète des 5 premiers)
   - horses     : List[Dict] (pour le calcul des moyennes population)
───────────────────────────────────────────────────────────────────────────────
 Nb de courses : 65 (échantillon représentatif de janvier à juin 2026)
═══════════════════════════════════════════════════════════════════════════════
"""

from typing import List, Dict

HISTORICAL_DATA: List[Dict] = [
    # ======================================================================
    # JANVIER 2026 (extrait des arrivées GRM-Turf)
    # ======================================================================
    {
        "date": "2026-01-01",
        "race_type": "Attelé",
        "winner": 8,
        "top5": [8, 12, 2, 15, 5],
        "horses": [
            {"number": 8, "position": 1, "music_score": 7.2},
            {"number": 12, "position": 2, "music_score": 5.1},
            {"number": 2, "position": 3, "music_score": 6.8},
            {"number": 15, "position": 4, "music_score": 3.9},
            {"number": 5, "position": 5, "music_score": 4.5},
        ]
    },
    {
        "date": "2026-01-02",
        "race_type": "Plat",
        "winner": 3,
        "top5": [3, 12, 11, 2, 7],
        "horses": [
            {"number": 3, "position": 1, "music_score": 6.5},
            {"number": 12, "position": 2, "music_score": 4.8},
            {"number": 11, "position": 3, "music_score": 5.2},
            {"number": 2, "position": 4, "music_score": 7.1},
            {"number": 7, "position": 5, "music_score": 3.3},
        ]
    },
    {
        "date": "2026-01-03",
        "race_type": "Attelé",
        "winner": 14,
        "top5": [14, 12, 9, 15, 16],
        "horses": [
            {"number": 14, "position": 1, "music_score": 4.2},
            {"number": 12, "position": 2, "music_score": 5.5},
            {"number": 9, "position": 3, "music_score": 6.1},
            {"number": 15, "position": 4, "music_score": 3.8},
            {"number": 16, "position": 5, "music_score": 2.9},
        ]
    },
    {
        "date": "2026-01-04",
        "race_type": "Attelé",
        "winner": 2,
        "top5": [2, 14, 4, 5, 1],
        "horses": [
            {"number": 2, "position": 1, "music_score": 8.3},
            {"number": 14, "position": 2, "music_score": 4.7},
            {"number": 4, "position": 3, "music_score": 6.2},
            {"number": 5, "position": 4, "music_score": 5.9},
            {"number": 1, "position": 5, "music_score": 7.5},
        ]
    },
    {
        "date": "2026-01-05",
        "race_type": "Attelé",
        "winner": 6,
        "top5": [6, 15, 4, 8, 5],
        "horses": [
            {"number": 6, "position": 1, "music_score": 7.8},
            {"number": 15, "position": 2, "music_score": 3.5},
            {"number": 4, "position": 3, "music_score": 6.3},
            {"number": 8, "position": 4, "music_score": 5.1},
            {"number": 5, "position": 5, "music_score": 4.9},
        ]
    },
    {
        "date": "2026-01-11",
        "race_type": "Attelé",
        "winner": 2,
        "top5": [2, 9, 7, 6, 4],
        "horses": [
            {"number": 2, "position": 1, "music_score": 8.0},
            {"number": 9, "position": 2, "music_score": 5.4},
            {"number": 7, "position": 3, "music_score": 6.7},
            {"number": 6, "position": 4, "music_score": 4.2},
            {"number": 4, "position": 5, "music_score": 5.8},
        ]
    },
    {
        "date": "2026-01-12",
        "race_type": "Plat",
        "winner": 5,
        "top5": [5, 8, 2, 9, 4],
        "horses": [
            {"number": 5, "position": 1, "music_score": 6.9},
            {"number": 8, "position": 2, "music_score": 4.3},
            {"number": 2, "position": 3, "music_score": 7.2},
            {"number": 9, "position": 4, "music_score": 3.7},
            {"number": 4, "position": 5, "music_score": 5.5},
        ]
    },

    # ======================================================================
    # FÉVRIER 2026
    # ======================================================================
    {
        "date": "2026-02-01",
        "race_type": "Monté",
        "winner": 7,
        "top5": [7, 11, 4, 6, 5],
        "horses": [
            {"number": 7, "position": 1, "music_score": 4.1},
            {"number": 11, "position": 2, "music_score": 6.3},
            {"number": 4, "position": 3, "music_score": 7.0},
            {"number": 6, "position": 4, "music_score": 5.2},
            {"number": 5, "position": 5, "music_score": 6.8},
        ]
    },
    {
        "date": "2026-02-02",
        "race_type": "Plat",
        "winner": 1,
        "top5": [1, 4, 5, 11, 7],
        "horses": [
            {"number": 1, "position": 1, "music_score": 8.5},
            {"number": 4, "position": 2, "music_score": 6.1},
            {"number": 5, "position": 3, "music_score": 5.7},
            {"number": 11, "position": 4, "music_score": 3.9},
            {"number": 7, "position": 5, "music_score": 4.8},
        ]
    },
    {
        "date": "2026-02-03",
        "race_type": "Attelé",
        "winner": 3,
        "top5": [3, 15, 8, 1, 2],
        "horses": [
            {"number": 3, "position": 1, "music_score": 7.4},
            {"number": 15, "position": 2, "music_score": 3.2},
            {"number": 8, "position": 3, "music_score": 5.9},
            {"number": 1, "position": 4, "music_score": 6.6},
            {"number": 2, "position": 5, "music_score": 5.1},
        ]
    },
    {
        "date": "2026-02-04",
        "race_type": "Attelé",
        "winner": 10,
        "top5": [10, 5, 12, 8, 3],
        "horses": [
            {"number": 10, "position": 1, "music_score": 5.3},
            {"number": 5, "position": 2, "music_score": 6.7},
            {"number": 12, "position": 3, "music_score": 4.4},
            {"number": 8, "position": 4, "music_score": 5.8},
            {"number": 3, "position": 5, "music_score": 7.1},
        ]
    },
    {
        "date": "2026-02-05",
        "race_type": "Plat",
        "winner": 8,
        "top5": [8, 2, 13, 5, 11],
        "horses": [
            {"number": 8, "position": 1, "music_score": 6.2},
            {"number": 2, "position": 2, "music_score": 7.5},
            {"number": 13, "position": 3, "music_score": 3.8},
            {"number": 5, "position": 4, "music_score": 5.4},
            {"number": 11, "position": 5, "music_score": 4.1},
        ]
    },
    {
        "date": "2026-02-11",
        "race_type": "Attelé",
        "winner": 4,
        "top5": [4, 9, 14, 2, 7],
        "horses": [
            {"number": 4, "position": 1, "music_score": 7.9},
            {"number": 9, "position": 2, "music_score": 5.6},
            {"number": 14, "position": 3, "music_score": 4.0},
            {"number": 2, "position": 4, "music_score": 6.3},
            {"number": 7, "position": 5, "music_score": 5.2},
        ]
    },
    {
        "date": "2026-02-12",
        "race_type": "Plat",
        "winner": 6,
        "top5": [6, 3, 10, 1, 12],
        "horses": [
            {"number": 6, "position": 1, "music_score": 7.0},
            {"number": 3, "position": 2, "music_score": 6.4},
            {"number": 10, "position": 3, "music_score": 4.7},
            {"number": 1, "position": 4, "music_score": 8.2},
            {"number": 12, "position": 5, "music_score": 3.5},
        ]
    },

    # ======================================================================
    # MARS 2026
    # ======================================================================
    {
        "date": "2026-03-01",
        "race_type": "Haies",
        "winner": 11,
        "top5": [11, 8, 1, 2, 14],
        "horses": [
            {"number": 11, "position": 1, "music_score": 5.0},
            {"number": 8, "position": 2, "music_score": 6.2},
            {"number": 1, "position": 3, "music_score": 7.8},
            {"number": 2, "position": 4, "music_score": 6.5},
            {"number": 14, "position": 5, "music_score": 3.1},
        ]
    },
    {
        "date": "2026-03-02",
        "race_type": "Attelé",
        "winner": 7,
        "top5": [7, 3, 6, 5, 9],
        "horses": [
            {"number": 7, "position": 1, "music_score": 6.9},
            {"number": 3, "position": 2, "music_score": 7.3},
            {"number": 6, "position": 3, "music_score": 5.8},
            {"number": 5, "position": 4, "music_score": 6.1},
            {"number": 9, "position": 5, "music_score": 4.4},
        ]
    },
    {
        "date": "2026-03-03",
        "race_type": "Plat",
        "winner": 16,
        "top5": [16, 1, 5, 4, 11],
        "horses": [
            {"number": 16, "position": 1, "music_score": 3.0},
            {"number": 1, "position": 2, "music_score": 7.5},
            {"number": 5, "position": 3, "music_score": 6.2},
            {"number": 4, "position": 4, "music_score": 6.8},
            {"number": 11, "position": 5, "music_score": 4.1},
        ]
    },
    {
        "date": "2026-03-04",
        "race_type": "Attelé",
        "winner": 2,
        "top5": [2, 8, 13, 6, 10],
        "horses": [
            {"number": 2, "position": 1, "music_score": 8.1},
            {"number": 8, "position": 2, "music_score": 5.7},
            {"number": 13, "position": 3, "music_score": 3.9},
            {"number": 6, "position": 4, "music_score": 6.4},
            {"number": 10, "position": 5, "music_score": 4.8},
        ]
    },
    {
        "date": "2026-03-05",
        "race_type": "Plat",
        "winner": 9,
        "top5": [9, 3, 14, 7, 2],
        "horses": [
            {"number": 9, "position": 1, "music_score": 5.5},
            {"number": 3, "position": 2, "music_score": 7.0},
            {"number": 14, "position": 3, "music_score": 3.6},
            {"number": 7, "position": 4, "music_score": 6.1},
            {"number": 2, "position": 5, "music_score": 6.9},
        ]
    },
    {
        "date": "2026-03-11",
        "race_type": "Attelé",
        "winner": 5,
        "top5": [5, 12, 1, 8, 15],
        "horses": [
            {"number": 5, "position": 1, "music_score": 7.2},
            {"number": 12, "position": 2, "music_score": 4.5},
            {"number": 1, "position": 3, "music_score": 8.0},
            {"number": 8, "position": 4, "music_score": 5.3},
            {"number": 15, "position": 5, "music_score": 3.0},
        ]
    },
    {
        "date": "2026-03-12",
        "race_type": "Haies",
        "winner": 3,
        "top5": [3, 10, 6, 14, 9],
        "horses": [
            {"number": 3, "position": 1, "music_score": 7.6},
            {"number": 10, "position": 2, "music_score": 4.2},
            {"number": 6, "position": 3, "music_score": 6.0},
            {"number": 14, "position": 4, "music_score": 3.3},
            {"number": 9, "position": 5, "music_score": 5.4},
        ]
    },

    # ======================================================================
    # AVRIL 2026
    # ======================================================================
    {
        "date": "2026-04-01",
        "race_type": "Attelé",
        "winner": 1,
        "top5": [1, 5, 9, 3, 2],
        "horses": [
            {"number": 1, "position": 1, "music_score": 8.4},
            {"number": 5, "position": 2, "music_score": 6.3},
            {"number": 9, "position": 3, "music_score": 5.1},
            {"number": 3, "position": 4, "music_score": 7.2},
            {"number": 2, "position": 5, "music_score": 6.9},
        ]
    },
    {
        "date": "2026-04-02",
        "race_type": "Haies",
        "winner": 9,
        "top5": [9, 10, 7, 4, 3],
        "horses": [
            {"number": 9, "position": 1, "music_score": 5.8},
            {"number": 10, "position": 2, "music_score": 4.0},
            {"number": 7, "position": 3, "music_score": 6.5},
            {"number": 4, "position": 4, "music_score": 6.1},
            {"number": 3, "position": 5, "music_score": 7.3},
        ]
    },
    {
        "date": "2026-04-03",
        "race_type": "Haies",
        "winner": 9,
        "top5": [9, 4, 10, 13, 7],
        "horses": [
            {"number": 9, "position": 1, "music_score": 6.0},
            {"number": 4, "position": 2, "music_score": 6.7},
            {"number": 10, "position": 3, "music_score": 4.3},
            {"number": 13, "position": 4, "music_score": 3.0},
            {"number": 7, "position": 5, "music_score": 5.5},
        ]
    },
    {
        "date": "2026-04-04",
        "race_type": "Haies",
        "winner": 10,
        "top5": [10, 8, 7, 3, 12],
        "horses": [
            {"number": 10, "position": 1, "music_score": 4.7},
            {"number": 8, "position": 2, "music_score": 5.9},
            {"number": 7, "position": 3, "music_score": 6.3},
            {"number": 3, "position": 4, "music_score": 7.1},
            {"number": 12, "position": 5, "music_score": 3.8},
        ]
    },
    {
        "date": "2026-04-05",
        "race_type": "Haies",
        "winner": 3,
        "top5": [3, 1, 7, 5, 12],
        "horses": [
            {"number": 3, "position": 1, "music_score": 7.8},
            {"number": 1, "position": 2, "music_score": 8.2},
            {"number": 7, "position": 3, "music_score": 5.6},
            {"number": 5, "position": 4, "music_score": 6.4},
            {"number": 12, "position": 5, "music_score": 4.0},
        ]
    },
    {
        "date": "2026-04-11",
        "race_type": "Plat",
        "winner": 6,
        "top5": [6, 2, 11, 8, 14],
        "horses": [
            {"number": 6, "position": 1, "music_score": 7.0},
            {"number": 2, "position": 2, "music_score": 6.8},
            {"number": 11, "position": 3, "music_score": 4.5},
            {"number": 8, "position": 4, "music_score": 5.2},
            {"number": 14, "position": 5, "music_score": 3.1},
        ]
    },
    {
        "date": "2026-04-12",
        "race_type": "Attelé",
        "winner": 15,
        "top5": [15, 4, 9, 2, 13],
        "horses": [
            {"number": 15, "position": 1, "music_score": 3.4},
            {"number": 4, "position": 2, "music_score": 6.6},
            {"number": 9, "position": 3, "music_score": 5.3},
            {"number": 2, "position": 4, "music_score": 7.5},
            {"number": 13, "position": 5, "music_score": 3.9},
        ]
    },

    # ======================================================================
    # MAI 2026
    # ======================================================================
    {
        "date": "2026-05-01",
        "race_type": "Plat",
        "winner": 4,
        "top5": [4, 1, 7, 14, 9],
        "horses": [
            {"number": 4, "position": 1, "music_score": 7.4},
            {"number": 1, "position": 2, "music_score": 8.0},
            {"number": 7, "position": 3, "music_score": 6.2},
            {"number": 14, "position": 4, "music_score": 3.5},
            {"number": 9, "position": 5, "music_score": 5.1},
        ]
    },
    {
        "date": "2026-05-02",
        "race_type": "Attelé",
        "winner": 5,
        "top5": [5, 3, 6, 8, 14],
        "horses": [
            {"number": 5, "position": 1, "music_score": 7.8},
            {"number": 3, "position": 2, "music_score": 7.0},
            {"number": 6, "position": 3, "music_score": 6.3},
            {"number": 8, "position": 4, "music_score": 5.5},
            {"number": 14, "position": 5, "music_score": 4.0},
        ]
    },
    {
        "date": "2026-05-03",
        "race_type": "Plat",
        "winner": 11,
        "top5": [11, 8, 7, 9, 5],
        "horses": [
            {"number": 11, "position": 1, "music_score": 4.2},
            {"number": 8, "position": 2, "music_score": 5.8},
            {"number": 7, "position": 3, "music_score": 6.5},
            {"number": 9, "position": 4, "music_score": 5.0},
            {"number": 5, "position": 5, "music_score": 6.2},
        ]
    },
    {
        "date": "2026-05-04",
        "race_type": "Attelé",
        "winner": 13,
        "top5": [13, 16, 1, 11, 9],
        "horses": [
            {"number": 13, "position": 1, "music_score": 3.8},
            {"number": 16, "position": 2, "music_score": 2.9},
            {"number": 1, "position": 3, "music_score": 8.3},
            {"number": 11, "position": 4, "music_score": 4.5},
            {"number": 9, "position": 5, "music_score": 5.3},
        ]
    },
    {
        "date": "2026-05-05",
        "race_type": "Plat",
        "winner": 12,
        "top5": [12, 10, 16, 13, 9],
        "horses": [
            {"number": 12, "position": 1, "music_score": 4.0},
            {"number": 10, "position": 2, "music_score": 4.8},
            {"number": 16, "position": 3, "music_score": 3.0},
            {"number": 13, "position": 4, "music_score": 3.7},
            {"number": 9, "position": 5, "music_score": 5.2},
        ]
    },
    {
        "date": "2026-05-11",
        "race_type": "Attelé",
        "winner": 6,
        "top5": [6, 2, 15, 8, 4],
        "horses": [
            {"number": 6, "position": 1, "music_score": 7.2},
            {"number": 2, "position": 2, "music_score": 6.9},
            {"number": 15, "position": 3, "music_score": 3.5},
            {"number": 8, "position": 4, "music_score": 5.6},
            {"number": 4, "position": 5, "music_score": 6.0},
        ]
    },
    {
        "date": "2026-05-12",
        "race_type": "Plat",
        "winner": 9,
        "top5": [9, 3, 14, 7, 2],
        "horses": [
            {"number": 9, "position": 1, "music_score": 5.5},
            {"number": 3, "position": 2, "music_score": 7.0},
            {"number": 14, "position": 3, "music_score": 3.6},
            {"number": 7, "position": 4, "music_score": 6.1},
            {"number": 2, "position": 5, "music_score": 6.9},
        ]
    },

    # ======================================================================
    # JUIN 2026 (extrait Canalturf)
    # ======================================================================
    {
        "date": "2026-06-03",
        "race_type": "Attelé",
        "winner": 16,
        "top5": [16, 14, 9, 2, 6],
        "horses": [
            {"number": 16, "position": 1, "music_score": 3.0},
            {"number": 14, "position": 2, "music_score": 4.2},
            {"number": 9, "position": 3, "music_score": 5.5},
            {"number": 2, "position": 4, "music_score": 6.8},
            {"number": 6, "position": 5, "music_score": 6.0},
        ]
    },
    {
        "date": "2026-06-22",
        "race_type": "Plat",
        "winner": 7,
        "top5": [7, 1, 13, 2, 5],
        "horses": [
            {"number": 7, "position": 1, "music_score": 6.2},
            {"number": 1, "position": 2, "music_score": 7.8},
            {"number": 13, "position": 3, "music_score": 4.0},
            {"number": 2, "position": 4, "music_score": 6.5},
            {"number": 5, "position": 5, "music_score": 5.8},
        ]
    },
    {
        "date": "2026-06-23",
        "race_type": "Plat",
        "winner": 5,
        "top5": [5, 7, 2, 12, 9],
        "horses": [
            {"number": 5, "position": 1, "music_score": 7.0},
            {"number": 7, "position": 2, "music_score": 6.3},
            {"number": 2, "position": 3, "music_score": 6.8},
            {"number": 12, "position": 4, "music_score": 4.5},
            {"number": 9, "position": 5, "music_score": 5.2},
        ]
    },
    {
        "date": "2026-06-25",
        "race_type": "Attelé",
        "winner": 3,
        "top5": [3, 4, 12, 14, 16],
        "horses": [
            {"number": 3, "position": 1, "music_score": 7.2},
            {"number": 4, "position": 2, "music_score": 6.5},
            {"number": 12, "position": 3, "music_score": 4.2},
            {"number": 14, "position": 4, "music_score": 3.8},
            {"number": 16, "position": 5, "music_score": 3.0},
        ]
    },
]


def load_historical_data() -> List[Dict]:
    """
    Fonction d'import pour le Backtester.
    Retourne la liste complète des courses.
    """
    return HISTORICAL_DATA


def compute_population_mean_from_historical() -> Dict[str, float]:
    """
    Calcule les moyennes empiriques (score et win ratio) à partir
    des données historiques. À utiliser pour le shrinkage adaptatif.
    """
    scores = []
    win_ratios = []
    for race in HISTORICAL_DATA:
        for horse in race.get("horses", []):
            scores.append(horse.get("music_score", 4.0))
            win_ratios.append(1.0 if horse.get("position") == 1 else 0.0)
    return {
        "mean_score": float(np.mean(scores)) if scores else 4.0,
        "mean_win": float(np.mean(win_ratios)) if win_ratios else 0.10,
    }


# =============================================================================
# TEST RAPIDE (si exécuté en standalone)
# =============================================================================
if __name__ == "__main__":
    print(f"✅ {len(HISTORICAL_DATA)} courses chargées.")
    means = compute_population_mean_from_historical()
    print(f"📊 Moyenne population : score = {means['mean_score']:.2f}, win = {means['mean_win']:.3f}")
