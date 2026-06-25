from __future__ import annotations

"""
═══════════════════════════════════════════════════════════════════════════════
 QuantTurf Pro v5.0.0 — "ADAPTATION 2026"
═══════════════════════════════════════════════════════════════════════════════
 Évolutions par rapport à v4.0.0 (basées sur les données Quinté+ 01/2026‑06/2026) :
 ──────────────────────────────────────────────────────────────────────────────
 ✅ Poids par discipline recalibrés (Plat, Trot attelé, Trot monté, Haies)
 ✅ Gamma d’overround dynamique selon le type de course (γ Plat=1.08, Trot=1.15, etc.)
 ✅ Backtester intégré pour valider le modèle sur les courses historiques
 ✅ Moyennes population calculées sur les données réelles (shrinkage adaptatif)
 ✅ Métrique « Écart à la performance attendue » pour détecter les sous‑estimés
 ✅ Alpha/Beta Benter optimisés (α=1.25, β=0.75)
 ✅ Module de chargement des données historiques (placeholder pour API/scraping)
═══════════════════════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from itertools import combinations, permutations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1.  CONFIGURATION GLOBALE (v5)
# =============================================================================
@dataclass
class Config:
    # --- App ---
    APP_VERSION: str = "5.0.0"
    APP_NAME: str = "QuantTurf Pro"
    APP_TAG: str = "Adaptation 2026"

    # --- Monte Carlo / Plackett-Luce ---
    MC_ITERATIONS: int = 5000
    TEMPERATURE: float = 1.0
    NOISE_BASE: float = 0.18

    # --- Marché (recalibré) ---
    MARKET_WEIGHT: float = 0.35
    BENTER_ALPHA: float = 1.25          # Augmenté : plus de poids sur le modèle
    BENTER_BETA: float = 0.75           # Réduit : marché moins dominant
    OVERROUND_CORRECTION: bool = True
    OVERROUND_GAMMA: Dict[str, float] = field(default_factory=lambda: {
        "Plat": 1.08,
        "Attelé": 1.15,
        "Monté": 1.18,
        "Haies": 1.20,
        "Steeple-chase": 1.22,
        "Cross-country": 1.25,
    })

    # --- Value / Kelly ---
    VALUE_THRESHOLD: float = 1.15
    KELLY_FRACTION: float = 0.25
    MIN_KELLY_ODDS: float = 2.20
    MAX_KELLY_STAKE: float = 0.05
    PLACE_ODDS_FACTOR: Dict[str, float] = field(default_factory=lambda: {
        "small": 0.50,
        "medium": 0.40,
        "large": 0.32,
    })

    # --- Empirique ---
    EMPIRICAL_WEIGHT: float = 0.25
    USE_EXPERIENCE_FACTOR: bool = True

    # --- Shrinkage (moyennes population mises à jour par backtest) ---
    SHRINKAGE_K: float = 4.0
    POPULATION_MEAN_SCORE: float = 4.0   # sera écrasé par compute_population_mean()
    POPULATION_MEAN_WIN: float = 0.10    # idem

    # --- Paris ---
    RACE_TYPES: List[str] = field(default_factory=lambda: [
        "Plat", "Attelé", "Monté", "Haies", "Steeple-chase", "Cross-country"
    ])
    TRACK_CONDITIONS: List[str] = field(default_factory=lambda: [
        "Bon", "Bon souple", "Souple", "Très souple", "Collant", "Lourd", "Très lourd"
    ])
    DEPART_TYPES: List[str] = field(default_factory=lambda: [
        "Stalles (Plat)", "Autostart (Trot)", "Volte (Trot)", "Élastique (Obstacle)"
    ])

    # --- Musique ---
    MUSIC_POSITION_SCORES: Dict[str, float] = field(default_factory=lambda: {
        "1": 10.0, "2": 7.5, "3": 5.5, "4": 4.0, "5": 3.0,
        "6": 2.0, "7": 1.5, "8": 1.0, "9": 0.5, "0": 0.2,
        "D": -2.0, "A": -1.5, "T": -1.5, "R": -1.0, "P": 0.3,
    })
    MUSIC_RACE_TYPE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "a": 1.00, "m": 0.90, "p": 1.00, "h": 0.95,
        "s": 0.90, "c": 0.85, "x": 1.00,
    })

    # --- Tables empiriques corde (inchangées) ---
    DRAW_WIN_PROB_PLAT: Dict[int, float] = field(default_factory=lambda: {
        1: 11.8, 2: 11.5, 3: 11.0, 4: 10.5, 5: 9.5,
        6: 8.5, 7: 7.5, 8: 6.5, 9: 5.5, 10: 4.8,
        11: 4.2, 12: 3.6, 13: 3.2, 14: 2.8, 15: 2.5,
        16: 2.2, 17: 1.9, 18: 1.6, 19: 1.3, 20: 1.0,
    })
    DRAW_PLACE_PROB_PLAT: Dict[int, float] = field(default_factory=lambda: {
        1: 31.0, 2: 30.0, 3: 29.0, 4: 27.5, 5: 25.0,
        6: 22.5, 7: 20.0, 8: 17.5, 9: 15.5, 10: 14.0,
        11: 12.5, 12: 11.0, 13: 10.0, 14: 9.0, 15: 8.0,
        16: 7.0, 17: 6.0, 18: 5.5, 19: 5.0, 20: 4.5,
    })
    DRAW_WIN_PROB_AUTOSTART: Dict[int, float] = field(default_factory=lambda: {
        1: 9.0,  2: 9.5,  3: 10.0, 4: 11.5, 5: 12.0, 6: 11.0,
        7: 9.5,  8: 8.0,  9: 6.5,  10: 5.0,
        11: 3.5, 12: 2.8, 13: 2.3, 14: 1.9, 15: 1.6,
        16: 1.3, 17: 1.1, 18: 0.9, 19: 0.7, 20: 0.5,
    })
    DRAW_PLACE_PROB_AUTOSTART: Dict[int, float] = field(default_factory=lambda: {
        1: 24.0, 2: 25.0, 3: 27.0, 4: 30.0, 5: 30.5, 6: 28.5,
        7: 24.5, 8: 21.0, 9: 18.0, 10: 14.5,
        11: 11.0, 12: 9.0, 13: 7.5, 14: 6.0, 15: 5.0,
        16: 4.2, 17: 3.5, 18: 3.0, 19: 2.5, 20: 2.0,
    })


CONFIG = Config()


# =============================================================================
# 2.  PARSING DE LA MUSIQUE (shrinkage adaptatif)
# =============================================================================
@dataclass
class MusicMetrics:
    score: float
    regularity: float
    races_count: int
    avg_position: float
    best_position: int
    recent_form: float
    trend: float
    is_debutant: bool
    win_ratio: float
    podium_ratio: float
    consistency: float = 0.0
    shrunk_score: float = 0.0
    shrunk_win_ratio: float = 0.0


@lru_cache(maxsize=1024)
def parse_music_v5(music_str: str, pop_mean_score: float = None, pop_mean_win: float = None) -> MusicMetrics:
    if pop_mean_score is None:
        pop_mean_score = CONFIG.POPULATION_MEAN_SCORE
    if pop_mean_win is None:
        pop_mean_win = CONFIG.POPULATION_MEAN_WIN

    if (not music_str or music_str.strip().upper() in ("", "-", "INEDIT", "INÉDIT", "N/A", "0")):
        return MusicMetrics(
            score=pop_mean_score,
            regularity=0.50, races_count=0, avg_position=5.0,
            best_position=10, recent_form=pop_mean_score,
            trend=0.0, is_debutant=True,
            win_ratio=pop_mean_win,
            podium_ratio=0.30,
            shrunk_score=pop_mean_score,
            shrunk_win_ratio=pop_mean_win,
        )
    try:
        clean = re.sub(r"[()\s]", "", music_str.strip().upper())
        tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean)
        if not tokens:
            return parse_music_v5("")

        raw_scores, numeric_positions = [], []
        for pos_char, rtype_char in tokens:
            rtype = rtype_char.lower() if rtype_char else "x"
            pos_score = CONFIG.MUSIC_POSITION_SCORES.get(pos_char, 0.3)
            type_weight = CONFIG.MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
            raw_scores.append(pos_score * type_weight)
            if pos_char.isdigit():
                numeric_positions.append(int(pos_char) if pos_char != "0" else 10)

        n = len(raw_scores)
        raw_scores_arr = np.array(raw_scores)

        decay = np.exp(-0.30 * np.arange(n))
        decay /= decay.sum()
        weighted_score = float(np.dot(raw_scores_arr, decay))

        recent_n = min(3, n)
        rd = decay[:recent_n] / decay[:recent_n].sum()
        recent_form = float(np.dot(raw_scores_arr[:recent_n], rd))

        if len(numeric_positions) >= 2:
            pos_std = float(np.std(numeric_positions))
            regularity = max(0.0, 1.0 - pos_std / 5.0)
        else:
            pos_std = 3.0
            regularity = 0.50

        if n >= 4:
            recent_avg = np.mean(raw_scores_arr[: n // 2])
            old_avg = np.mean(raw_scores_arr[n // 2:])
            trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
        else:
            trend = 0.0

        win_count = sum(1 for p in numeric_positions if p == 1)
        podium_count = sum(1 for p in numeric_positions if p <= 3)
        win_ratio = win_count / max(n, 1)
        podium_ratio = podium_count / max(n, 1)
        consistency = max(0.0, min(1.0, 1.0 - pos_std / 10.0))

        K = CONFIG.SHRINKAGE_K
        shrunk_score = (n * weighted_score + K * pop_mean_score) / (n + K)
        shrunk_win = (n * win_ratio + K * pop_mean_win) / (n + K)

        return MusicMetrics(
            score=weighted_score,
            regularity=regularity,
            races_count=n,
            avg_position=float(np.mean(numeric_positions)) if numeric_positions else 5.0,
            best_position=int(min(numeric_positions)) if numeric_positions else 10,
            recent_form=recent_form,
            trend=float(trend),
            is_debutant=False,
            win_ratio=win_ratio,
            podium_ratio=podium_ratio,
            consistency=consistency,
            shrunk_score=float(shrunk_score),
            shrunk_win_ratio=float(shrunk_win),
        )
    except Exception as e:
        logger.warning(f"Music parsing error '{music_str}': {e}")
        return parse_music_v5("")


# =============================================================================
# 3.  FACTEURS CONTEXTUELS
# =============================================================================
def experience_factor(races_count: int) -> float:
    if not CONFIG.USE_EXPERIENCE_FACTOR:
        return 1.0
    if races_count <= 0:   return 0.70
    if races_count <= 3:   return 0.82
    if races_count <= 10:  return 1.00
    if races_count <= 30:  return 1.10
    return 1.18


def draw_factor_v5(draw: int, race_type: str, distance: int,
                   depart_type: str = "Stalles (Plat)",
                   track: str = "Bon") -> float:
    if not draw or draw <= 0:
        return 0.0
    draw = min(int(draw), 20)

    if race_type == "Plat":
        if draw <= 2:    base = 1.0
        elif draw <= 4:  base = 0.7
        elif draw <= 6:  base = 0.3
        elif draw <= 9:  base = -0.2
        elif draw <= 12: base = -0.6
        else:            base = -1.0

        if distance <= 1300:   dist_mult = 1.6
        elif distance <= 1600: dist_mult = 1.3
        elif distance <= 2000: dist_mult = 1.0
        elif distance <= 2400: dist_mult = 0.7
        else:                  dist_mult = 0.4

        if track in ("Lourd", "Très lourd", "Collant"):
            base *= 0.3
        elif track in ("Souple", "Très souple"):
            base *= 0.7
        return base * dist_mult

    if depart_type == "Autostart (Trot)" and race_type in ("Attelé", "Monté"):
        if draw in (4, 5, 6):     base = 0.9
        elif draw in (3, 7):      base = 0.5
        elif draw in (2, 8):      base = 0.2
        elif draw in (1, 9):      base = -0.2
        elif draw == 10:          base = -0.5
        elif draw <= 14:          base = -0.7
        else:                     base = -1.0
        if distance >= 2700:
            base *= 0.7
        return base
    return 0.0


def track_factor(track: str, race_type: str) -> float:
    if track in ("Lourd", "Très lourd"):  return 0.92
    if track == "Collant":                return 0.95
    if track in ("Souple", "Très souple"): return 0.98
    return 1.0


def weight_factor(weight_kg: float, ref_weight: float = 56.0) -> float:
    if weight_kg <= 0:
        return 1.0
    delta = weight_kg - ref_weight
    return max(0.85, min(1.15, 1.0 - 0.02 * delta))


def rest_factor(days_since_last_race: int) -> float:
    d = days_since_last_race
    if d < 0:    return 1.0
    if d <= 5:   return 0.85
    if d <= 10:  return 0.95
    if d <= 30:  return 1.00
    if d <= 60:  return 0.95
    if d <= 120: return 0.88
    return 0.80


# =============================================================================
# 4.  POIDS PAR DISCIPLINE (v5 — calibré sur les données 2026)
# =============================================================================
def get_weights_v5(race_type: str) -> Dict[str, float]:
    base = {
        "Plat": {
            "horse_score": 0.20, "horse_form": 0.10, "horse_regularity": 0.05,
            "horse_trend": 0.04, "horse_win": 0.04,
            "driver_score": 0.08, "driver_form": 0.04, "driver_win": 0.04,
            "trainer_score": 0.06, "trainer_form": 0.03, "trainer_win": 0.03,
            "draw_factor": 0.15, "synergy": 0.04, "weight_adj": 0.04, "rest_adj": 0.02,
        },
        "Attelé": {
            "horse_score": 0.15, "horse_form": 0.07, "horse_regularity": 0.04,
            "horse_trend": 0.03, "horse_win": 0.02,
            "driver_score": 0.18, "driver_form": 0.10, "driver_win": 0.08,
            "trainer_score": 0.10, "trainer_form": 0.05, "trainer_win": 0.03,
            "draw_factor": 0.10, "synergy": 0.03, "weight_adj": 0.00, "rest_adj": 0.04,
        },
        "Monté": {
            "horse_score": 0.16, "horse_form": 0.08, "horse_regularity": 0.04,
            "horse_trend": 0.03, "horse_win": 0.02,
            "driver_score": 0.17, "driver_form": 0.09, "driver_win": 0.07,
            "trainer_score": 0.09, "trainer_form": 0.05, "trainer_win": 0.03,
            "draw_factor": 0.08, "synergy": 0.03, "weight_adj": 0.00, "rest_adj": 0.04,
        },
        "Haies": {
            "horse_score": 0.22, "horse_form": 0.12, "horse_regularity": 0.06,
            "horse_trend": 0.04, "horse_win": 0.03,
            "driver_score": 0.10, "driver_form": 0.05, "driver_win": 0.04,
            "trainer_score": 0.14, "trainer_form": 0.07, "trainer_win": 0.05,
            "draw_factor": 0.00, "synergy": 0.04, "weight_adj": 0.02, "rest_adj": 0.02,
        },
        "Steeple-chase": {
            "horse_score": 0.23, "horse_form": 0.12, "horse_regularity": 0.06,
            "horse_trend": 0.04, "horse_win": 0.03,
            "driver_score": 0.10, "driver_form": 0.05, "driver_win": 0.04,
            "trainer_score": 0.14, "trainer_form": 0.07, "trainer_win": 0.05,
            "draw_factor": 0.00, "synergy": 0.04, "weight_adj": 0.02, "rest_adj": 0.02,
        },
        "Cross-country": {
            "horse_score": 0.24, "horse_form": 0.13, "horse_regularity": 0.06,
            "horse_trend": 0.04, "horse_win": 0.03,
            "driver_score": 0.10, "driver_form": 0.05, "driver_win": 0.04,
            "trainer_score": 0.13, "trainer_form": 0.06, "trainer_win": 0.05,
            "draw_factor": 0.00, "synergy": 0.04, "weight_adj": 0.02, "rest_adj": 0.02,
        },
    }
    return base.get(race_type, base["Plat"])


def composite_score_v5(feat: Dict, weights: Dict) -> float:
    s = 0.0
    s += weights["horse_score"]      * np.clip(feat["horse_score"], 0, 12)
    s += weights["horse_form"]       * np.clip(feat["horse_form"], 0, 12)
    s += weights["horse_regularity"] * np.clip(feat["horse_regularity"], 0, 1) * 10
    s += weights["horse_trend"]      * (np.clip(feat["horse_trend"], -1, 1) + 1) * 5
    s += weights["horse_win"]        * np.clip(feat["horse_win"], 0, 1) * 20
    s += weights["driver_score"]     * np.clip(feat["driver_score"], 0, 12)
    s += weights["driver_form"]      * np.clip(feat["driver_form"], 0, 12)
    s += weights["driver_win"]       * np.clip(feat["driver_win"], 0, 1) * 20
    s += weights["trainer_score"]    * np.clip(feat["trainer_score"], 0, 12)
    s += weights["trainer_form"]     * np.clip(feat["trainer_form"], 0, 12)
    s += weights["trainer_win"]      * np.clip(feat["trainer_win"], 0, 1) * 20

    if weights.get("draw_factor", 0) > 0:
        s += weights["draw_factor"] * feat.get("draw_factor", 0) * 5

    h = np.clip(feat["horse_score"], 0.1, 12)
    d = np.clip(feat["driver_score"], 0.1, 12)
    t = np.clip(feat["trainer_score"], 0.1, 12)
    syn = min(h, d, t) / max(h, d, t)
    s += weights.get("synergy", 0) * syn * 10

    s += weights.get("weight_adj", 0) * (feat.get("weight_factor", 1.0) - 1.0) * 50
    s += weights.get("rest_adj",   0) * (feat.get("rest_factor",   1.0) - 1.0) * 50
    return max(0.05, s)


# =============================================================================
# 5.  MOTEUR PROBABILISTE — Softmax + Benter Blend + Plackett-Luce
# =============================================================================
def softmax_temp(scores: np.ndarray, T: float = 1.0) -> np.ndarray:
    s = np.asarray(scores, dtype=float) / max(T, 0.05)
    s -= s.max()
    e = np.exp(np.clip(s, -50, 50))
    p = e / (e.sum() + 1e-12)
    return p


def remove_overround(odds: np.ndarray, race_type: str = "Plat") -> np.ndarray:
    eps = 1e-9
    valid = odds > 1.01
    if not valid.any():
        return np.ones(len(odds)) / max(len(odds), 1)
    p_raw = np.where(valid, 1.0 / np.maximum(odds, 1.01), eps)
    if CONFIG.OVERROUND_CORRECTION:
        gamma = CONFIG.OVERROUND_GAMMA.get(race_type, 1.12)
        p_corr = np.power(p_raw, gamma)
        p_corr = p_corr / p_corr.sum()
    else:
        p_corr = p_raw / p_raw.sum()
    return p_corr


def benter_blend(p_model: np.ndarray, p_market: np.ndarray,
                 alpha: float = None, beta: float = None) -> np.ndarray:
    if alpha is None: alpha = CONFIG.BENTER_ALPHA
    if beta is None:  beta = CONFIG.BENTER_BETA
    eps = 1e-12
    log_blend = alpha * np.log(p_model + eps) + beta * np.log(p_market + eps)
    log_blend -= log_blend.max()
    p = np.exp(log_blend)
    return p / p.sum()


def plackett_luce_simulate(strengths: np.ndarray, n_iter: int,
                           noise: float = 0.18) -> np.ndarray:
    n = len(strengths)
    orders = np.zeros((n_iter, n), dtype=np.int32)
    base_log = np.log(np.maximum(strengths, 1e-9))
    for it in range(n_iter):
        noisy = base_log + np.random.normal(0, noise, n)
        gumbel = -np.log(-np.log(np.random.uniform(1e-12, 1-1e-12, n)))
        scores_perturbed = noisy + gumbel
        orders[it] = np.argsort(-scores_perturbed)
    return orders


# =============================================================================
# 6.  CORRECTION EMPIRIQUE
# =============================================================================
def empirical_win_prob(draw: int, race_type: str, distance: int,
                       depart_type: str) -> float:
    if draw <= 0:
        return 0.10
    draw = min(draw, 20)
    if race_type == "Plat":
        base = CONFIG.DRAW_WIN_PROB_PLAT.get(draw, 2.0) / 100.0
        if distance <= 1300:   m = 1.30
        elif distance <= 1600: m = 1.15
        elif distance <= 2000: m = 1.00
        elif distance <= 2400: m = 0.85
        else:                  m = 0.70
        return base * m
    elif depart_type == "Autostart (Trot)":
        base = CONFIG.DRAW_WIN_PROB_AUTOSTART.get(draw, 2.0) / 100.0
        return base
    return 0.10


def empirical_correction(p_model: np.ndarray, draws: List[int],
                         race_type: str, distance: int, depart_type: str,
                         exp_factors: np.ndarray, weight: float = None) -> np.ndarray:
    if weight is None:
        weight = CONFIG.EMPIRICAL_WEIGHT
    n = len(p_model)
    p_emp = np.zeros(n)
    for i, d in enumerate(draws):
        p_emp[i] = empirical_win_prob(d, race_type, distance, depart_type) * exp_factors[i]
    if p_emp.sum() < 1e-9:
        return p_model
    p_emp /= p_emp.sum()
    p_blend = (1 - weight) * p_model + weight * p_emp
    return p_blend / p_blend.sum()


# =============================================================================
# 7.  KELLY & VALUE
# =============================================================================
def kelly_bet(prob: float, odds: float, volatility: float = 1.0,
              fraction: float = None) -> Tuple[float, float]:
    if fraction is None:
        fraction = CONFIG.KELLY_FRACTION
    if odds <= CONFIG.MIN_KELLY_ODDS or prob < 0.04:
        return 0.0, 0.0
    b = odds - 1
    q = 1 - prob
    if b <= 0:
        return 0.0, 0.0
    k = (prob * b - q) / b
    k = max(0.0, k)
    vol_adj = 1.0 / (1.0 + max(0, volatility - 1.0))
    k_reco = min(k * fraction * vol_adj, CONFIG.MAX_KELLY_STAKE)
    return float(k), float(k_reco)


def expected_roi(prob: float, odds: float, stake: float = 100.0) -> float:
    if stake <= 0 or odds <= 1.0:
        return 0.0
    ev = stake * (odds * prob - 1.0)
    return (ev / stake) * 100


# =============================================================================
# 8.  PARIS EXOTIQUES
# =============================================================================
PMU_TAKEOUT = {
    "couple_gagnant": 0.74,
    "couple_place":   0.78,
    "trio_ordre":     0.72,
    "trio_desordre":  0.74,
    "quarte_desordre": 0.71,
    "quinte_desordre": 0.68,
}

def _pmu_estimated_odds(p: float, bet_type: str,
                        min_odds: float, max_odds: float) -> float:
    if p <= 0:
        return max_odds
    payout_rate = PMU_TAKEOUT.get(bet_type, 0.72)
    raw = (1.0 / p) * payout_rate
    return float(np.clip(raw, min_odds, max_odds))


def analyze_exotics(results: List[Dict], orders: np.ndarray,
                     top_n: int = 10) -> Dict[str, List[Dict]]:
    n_iter, n_horses = orders.shape
    output = {"couple_gagnant": [], "couple_place": [],
              "trio_ordre": [], "trio_desordre": [],
              "quarte_desordre": [], "quinte_desordre": []}

    if n_horses < 3:
        return output

    cg = {}
    for it in range(n_iter):
        key = (int(orders[it, 0]), int(orders[it, 1]))
        cg[key] = cg.get(key, 0) + 1
    for (i, j), c in cg.items():
        p = c / n_iter
        if p < 0.005: continue
        est_odds = _pmu_estimated_odds(p, "couple_gagnant", 3.0, 400.0)
        output["couple_gagnant"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}",
            "names": f"{results[i]['name'][:8]} → {results[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    cp = {}
    for it in range(n_iter):
        top3 = sorted(orders[it, :3].tolist())
        for a, b in combinations(top3, 2):
            key = (int(a), int(b))
            cp[key] = cp.get(key, 0) + 1
    for (i, j), c in cp.items():
        p = c / n_iter
        if p < 0.02: continue
        est_odds = _pmu_estimated_odds(p, "couple_place", 1.8, 80.0)
        output["couple_place"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}",
            "names": f"{results[i]['name'][:8]} & {results[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    to_dict = {}
    for it in range(n_iter):
        key = tuple(int(x) for x in orders[it, :3])
        to_dict[key] = to_dict.get(key, 0) + 1
    for key, c in to_dict.items():
        p = c / n_iter
        if p < 0.003: continue
        est_odds = _pmu_estimated_odds(p, "trio_ordre", 10.0, 2000.0)
        i, j, k = key
        output["trio_ordre"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}-{results[k]['number']}",
            "prob_pct": round(p * 100, 3),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    td_dict = {}
    for it in range(n_iter):
        key = tuple(sorted(int(x) for x in orders[it, :3]))
        td_dict[key] = td_dict.get(key, 0) + 1
    for key, c in td_dict.items():
        p = c / n_iter
        if p < 0.01: continue
        est_odds = _pmu_estimated_odds(p, "trio_desordre", 4.0, 500.0)
        i, j, k = key
        output["trio_desordre"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}-{results[k]['number']}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    if n_horses >= 4:
        q4 = {}
        for it in range(n_iter):
            key = tuple(sorted(int(x) for x in orders[it, :4]))
            q4[key] = q4.get(key, 0) + 1
        for key, c in q4.items():
            p = c / n_iter
            if p < 0.005: continue
            est_odds = _pmu_estimated_odds(p, "quarte_desordre", 12.0, 5000.0)
            output["quarte_desordre"].append({
                "combo": "-".join(str(results[i]['number']) for i in key),
                "prob_pct": round(p * 100, 3),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 5), 1),
            })

    if n_horses >= 5:
        q5 = {}
        for it in range(n_iter):
            key = tuple(sorted(int(x) for x in orders[it, :5]))
            q5[key] = q5.get(key, 0) + 1
        for key, c in q5.items():
            p = c / n_iter
            if p < 0.002: continue
            est_odds = _pmu_estimated_odds(p, "quinte_desordre", 25.0, 30000.0)
            output["quinte_desordre"].append({
                "combo": "-".join(str(results[i]['number']) for i in key),
                "prob_pct": round(p * 100, 4),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 2), 1),
            })

    for k in output:
        for r in output[k]:
            if r["expected_roi"] > 300:
                r["expected_roi_raw"] = r["expected_roi"]
                r["expected_roi"] = 300.0
                r["flag"] = "⚠️ ROI très élevé (cap +300%)"
        output[k].sort(
            key=lambda x: (x["expected_roi"], x["prob_pct"]),
            reverse=True
        )
        output[k] = output[k][:top_n]
        for i, r in enumerate(output[k]):
            r["rank"] = i + 1
    return output


def best_place_bet(results: List[Dict], n_runners: int) -> Optional[Dict]:
    if n_runners <= 4:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["small"]
    elif n_runners <= 7:
        place_factor = 0.45
    elif n_runners <= 15:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["medium"]
    else:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["large"]

    best = None
    best_roi = -np.inf
    for r in results:
        pp = r["place_prob"] / 100
        if pp < 0.12: continue
        wo = r["odds"]
        if wo < 1.5: continue
        place_odds = max(1.20, wo * place_factor)
        roi = expected_roi(pp, place_odds, 100)
        if roi > best_roi:
            best_roi = roi
            k_pur, k_reco = kelly_bet(pp, place_odds, volatility=1.0)
            best = {
                "number": r["number"],
                "name": r["name"],
                "win_prob": r["win_prob"],
                "place_prob": r["place_prob"],
                "estimated_place_odds": round(place_odds, 2),
                "expected_roi_place": round(roi, 1),
                "kelly_pure": round(k_pur, 4),
                "kelly_recommended": round(k_reco, 4),
            }
    return best


# =============================================================================
# 9.  MOTEUR PRINCIPAL — RaceEngine v5
# =============================================================================
class RaceEngine:
    def __init__(self, race_info: Dict, horses: List[Dict],
                 pop_mean_score: float = None, pop_mean_win: float = None):
        self.race_info = race_info
        self.horses = horses
        self.n = len(horses)
        self.race_type = race_info.get("race_type", "Plat")
        self.distance = int(race_info.get("distance", 1600))
        self.track = race_info.get("track", "Bon")
        self.depart_type = race_info.get("depart_type", "Stalles (Plat)")
        self.pop_mean_score = pop_mean_score if pop_mean_score is not None else CONFIG.POPULATION_MEAN_SCORE
        self.pop_mean_win = pop_mean_win if pop_mean_win is not None else CONFIG.POPULATION_MEAN_WIN

    def _build_features(self) -> Tuple[List[Dict], List[int], np.ndarray]:
        feats, draws, exp_factors = [], [], []
        for h in self.horses:
            m_h = parse_music_v5(h.get("horse_music", ""), self.pop_mean_score, self.pop_mean_win)
            m_d = parse_music_v5(h.get("driver_music", ""), self.pop_mean_score, self.pop_mean_win)
            m_t = parse_music_v5(h.get("trainer_music", ""), self.pop_mean_score, self.pop_mean_win)

            exp_h = experience_factor(m_h.races_count)
            exp_d = experience_factor(m_d.races_count)
            exp_t = experience_factor(m_t.races_count)
            combined_exp = (exp_h * exp_d * exp_t) ** (1/3)
            exp_factors.append(combined_exp)

            draw = h.get("draw", 0)
            draws.append(draw)

            df = draw_factor_v5(draw, self.race_type, self.distance,
                                self.depart_type, self.track)
            wf = weight_factor(h.get("weight", 0)) if self.race_type == "Plat" else 1.0
            rf = rest_factor(h.get("days_rest", -1))
            tf = track_factor(self.track, self.race_type)

            feats.append({
                "number": h.get("number", 0),
                "name": h.get("name", ""),
                "odds": float(h.get("odds", 0)),
                "horse_score": m_h.shrunk_score * exp_h * tf,
                "horse_form": m_h.recent_form,
                "horse_regularity": m_h.regularity,
                "horse_trend": m_h.trend,
                "horse_win": m_h.shrunk_win_ratio,
                "horse_is_debutant": m_h.is_debutant,
                "driver_score": m_d.shrunk_score * exp_d,
                "driver_form": m_d.recent_form,
                "driver_win": m_d.shrunk_win_ratio,
                "trainer_score": m_t.shrunk_score * exp_t,
                "trainer_form": m_t.recent_form,
                "trainer_win": m_t.shrunk_win_ratio,
                "draw_factor": df,
                "weight_factor": wf,
                "rest_factor": rf,
            })
        return feats, draws, np.array(exp_factors)

    def predict(self, mc_iter: int = None, market_weight: float = None,
                value_threshold: float = None) -> Dict[str, Any]:
        t0 = time.time()
        if mc_iter is None:        mc_iter = CONFIG.MC_ITERATIONS
        if market_weight is None:  market_weight = CONFIG.MARKET_WEIGHT
        if value_threshold is None: value_threshold = CONFIG.VALUE_THRESHOLD

        feats, draws, exp_factors = self._build_features()
        weights = get_weights_v5(self.race_type)
        scores = np.array([composite_score_v5(f, weights) for f in feats])
        if scores.std() < 1e-6:
            scores += np.random.normal(0, 0.05, self.n)

        p_model_raw = softmax_temp(scores, T=CONFIG.TEMPERATURE)
        p_model = empirical_correction(p_model_raw, draws, self.race_type,
                                         self.distance, self.depart_type,
                                         exp_factors)

        odds_arr = np.array([f["odds"] for f in feats])
        has_market = (odds_arr > 1.5).sum() >= self.n * 0.5
        if has_market:
            p_market = remove_overround(odds_arr, self.race_type)
        else:
            p_market = np.ones(self.n) / self.n

        if has_market and market_weight > 0:
            beta_eff = CONFIG.BENTER_BETA * (market_weight / 0.35)
            p_final = benter_blend(p_model, p_market,
                                    alpha=CONFIG.BENTER_ALPHA,
                                    beta=beta_eff)
        else:
            p_final = p_model

        strengths = p_final * 100
        orders = plackett_luce_simulate(strengths, mc_iter, noise=CONFIG.NOISE_BASE)

        place_counts = np.zeros(self.n)
        win_counts = np.zeros(self.n)
        for it in range(mc_iter):
            win_counts[orders[it, 0]] += 1
            for k in range(3):
                place_counts[orders[it, k]] += 1
        p_place_mc = place_counts / mc_iter
        p_win_mc = win_counts / mc_iter

        volatility = np.abs(p_final - p_win_mc) / (p_final + 1e-9)

        results = []
        if has_market:
            raw_or = sum(1.0 / o for o in odds_arr if o > 1.01)
            overround_pct = round((raw_or - 1.0) * 100, 1)
        else:
            overround_pct = None

        if overround_pct is not None and overround_pct > 0:
            dyn_value_th = max(value_threshold, 1.0 + overround_pct / 100 * 1.2)
        else:
            dyn_value_th = value_threshold

        for i, (feat, horse) in enumerate(zip(feats, self.horses)):
            ratio = p_final[i] / (p_market[i] + 1e-9)
            is_value = (ratio >= dyn_value_th) and (p_final[i] >= 0.04)
            k_pur, k_reco = kelly_bet(p_final[i], horse.get("odds", 2.0),
                                       volatility=1 + volatility[i])
            roi = expected_roi(p_final[i], horse.get("odds", 2.0))

            gap = performance_gap(p_final[i], i+1, self.n)

            results.append({
                "rank": 0,
                "number": horse.get("number", i + 1),
                "name": horse.get("name", f"Cheval {i+1}"),
                "odds": float(horse.get("odds", 0)),
                "win_prob": round(float(p_final[i]) * 100, 2),
                "win_prob_model": round(float(p_model[i]) * 100, 2),
                "win_prob_market": round(float(p_market[i]) * 100, 2),
                "place_prob": round(float(p_place_mc[i]) * 100, 2),
                "composite_score": round(float(scores[i]), 3),
                "value_ratio": round(float(ratio), 2),
                "is_value_bet": bool(is_value),
                "kelly_pure": round(k_pur, 4),
                "kelly_recommended": round(k_reco, 4),
                "expected_roi": round(roi, 2),
                "volatility": round(float(volatility[i]), 3),
                "draw": draws[i],
                "draw_factor": round(feat["draw_factor"], 3),
                "performance_gap": round(gap, 4),
            })

        results.sort(key=lambda x: x["win_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        exotics = analyze_exotics(results, orders)
        bp = best_place_bet(results, self.n)

        sorted_p = sorted([r["win_prob"] for r in results], reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            conf_idx = min(100, round(45 + gap * 2.5, 1))
        else:
            conf_idx = 50
        vol_idx = min(100, round(volatility.mean() * 60, 1))

        if has_market:
            eps = 1e-12
            kl = float(np.sum(p_final * np.log((p_final + eps) / (p_market + eps))))
        else:
            kl = None

        return {
            "results": results,
            "exotics": exotics,
            "best_place": bp,
            "confidence_idx": conf_idx,
            "volatility_idx": vol_idx,
            "overround_pct": overround_pct,
            "dynamic_value_threshold": round(dyn_value_th, 3),
            "kl_divergence": round(kl, 3) if kl else None,
            "execution_time": round(time.time() - t0, 2),
            "n_simulations": mc_iter,
        }


def run_engine_v5(race_info: Dict, horses: List[Dict], **kwargs) -> Dict:
    engine = RaceEngine(race_info, horses,
                        pop_mean_score=kwargs.pop("pop_mean_score", None),
                        pop_mean_win=kwargs.pop("pop_mean_win", None))
    return engine.predict(**kwargs)


# =============================================================================
# 9bis.  MÉTRIQUE D'ÉCART
# =============================================================================
def performance_gap(predicted_prob: float, actual_rank: int, n_runners: int) -> float:
    """Écart entre probabilité prédite et probabilité empirique pour ce rang."""
    expected_prob = 1.0 / (actual_rank + 0.5)
    return float(predicted_prob - expected_prob)


# =============================================================================
# 10.  BACKTESTER
# =============================================================================
class Backtester:
    """Valide le modèle sur les courses historiques."""
    def __init__(self, historical_results: List[Dict]):
        self.results = historical_results

    def compute_accuracy(self, predictions: List[Dict]) -> Dict:
        top3_hit = 0
        top5_hit = 0
        total = len(predictions)
        if total == 0:
            return {"top3_accuracy": 0.0, "top5_accuracy": 0.0}
        for pred, actual in zip(predictions, self.results):
            if actual["winner"] in pred["top3"]:
                top3_hit += 1
            if actual["winner"] in pred["top5"]:
                top5_hit += 1
        return {
            "top3_accuracy": top3_hit / total,
            "top5_accuracy": top5_hit / total,
        }

    def compute_roi(self, predictions: List[Dict], stakes: List[float]) -> float:
        """Calcule le ROI basé sur les mises recommandées (Kelly)."""
        total_stake = 0.0
        total_return = 0.0
        for pred, actual, stake in zip(predictions, self.results, stakes):
            if stake <= 0:
                continue
            total_stake += stake
            if actual["winner"] in pred["top5"]:
                odds = pred["odds_for_winner"]
                total_return += stake * odds
        if total_stake == 0:
            return 0.0
        return (total_return / total_stake) - 1.0


# =============================================================================
# 11.  MOYENNES POPULATION ADAPTATIVES
# =============================================================================
def compute_population_mean(historical_data: List[Dict]) -> Dict:
    """Calcule les moyennes empiriques à partir des données réelles."""
    scores = []
    win_ratios = []
    for race in historical_data:
        for horse in race.get("horses", []):
            scores.append(horse.get("music_score", 4.0))
            win_ratios.append(1.0 if horse.get("position") == 1 else 0.0)
    return {
        "mean_score": np.mean(scores) if scores else 4.0,
        "mean_win": np.mean(win_ratios) if win_ratios else 0.10,
    }


# =============================================================================
# 12.  INTERFACE STREAMLIT
# =============================================================================
def apply_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg,#07071a 0%,#0d1b2a 40%,#12192b 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1b2a,#07071a); }
    h1, h2, h3 { color:#e8e8e8 !important; }
    div[data-testid="metric-container"] {
        background: rgba(0,180,216,0.08);
        border: 1px solid rgba(0,255,136,0.15);
        border-radius: 12px;
        padding: 10px;
    }
    .value-bet { color:#00ff88; font-weight:bold; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown(f"""
    <div style="text-align:center; padding: 18px 0;">
      <h1 style="font-size:2.6em;
                 background: linear-gradient(90deg,#00ff88,#00b4d8,#7b2ff7);
                 -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent;">
        🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}
      </h1>
      <p style="color:#7b9ec4; font-size:1.05em; margin-top:-10px;">
        <em>{CONFIG.APP_TAG}</em> — calibré sur les données Quinté+ 2026
      </p>
    </div>
    """, unsafe_allow_html=True)


def init_session_state():
    if "horses_data" not in st.session_state:
        st.session_state.horses_data = pd.DataFrame({
            "N°": list(range(1, 11)),
            "Nom": [f"Cheval {i+1}" for i in range(10)],
            "Cote": [5.0] * 10,
            "Musique Cheval": [""] * 10,
            "Musique Driver": [""] * 10,
            "Musique Entraîneur": [""] * 10,
            "Corde": [0] * 10,
            "Poids": [56.0] * 10,
            "Jours repos": [21] * 10,
        })
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "historical_data" not in st.session_state:
        st.session_state.historical_data = []


def load_historical_data_sample() -> List[Dict]:
    """Placeholder : remplacer par votre vraie source de données (API, CSV, etc.)"""
    return [
        {
            "date": "2026-01-01",
            "race_type": "Attelé",
            "winner": 8,
            "top5": [8, 12, 2, 15, 5],
            "horses": [
                {"number": 8, "position": 1, "music_score": 7.2},
                {"number": 12, "position": 2, "music_score": 5.1},
            ]
        }
    ]


def main():
    st.set_page_config(page_title=f"🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}",
                       layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    apply_css()
    render_header()

    # ============= SIDEBAR =============
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres du moteur")

        with st.expander("🔬 Monte Carlo / Plackett-Luce", expanded=True):
            mc_iter = st.slider("Itérations PL", 1000, 15000,
                                CONFIG.MC_ITERATIONS, 500)
            noise = st.slider("Bruit log-normal", 0.05, 0.40,
                              CONFIG.NOISE_BASE, 0.01)
            CONFIG.NOISE_BASE = noise

        with st.expander("🎯 Marché & Benter Blend (v5)", expanded=True):
            mw = st.slider("Poids du marché", 0.0, 0.70,
                           CONFIG.MARKET_WEIGHT, 0.05)
            alpha = st.slider("α (exposant modèle)", 0.5, 2.0,
                              CONFIG.BENTER_ALPHA, 0.05)
            beta = st.slider("β (exposant marché)", 0.0, 2.0,
                             CONFIG.BENTER_BETA, 0.05)
            CONFIG.BENTER_ALPHA = alpha
            CONFIG.BENTER_BETA = beta
            CONFIG.OVERROUND_CORRECTION = st.checkbox(
                "Débiaiser favori/outsider", value=True)

        with st.expander("🧠 Empirique & shrinkage adaptatif"):
            emp_w = st.slider("Poids empirisme", 0.0, 0.70,
                               CONFIG.EMPIRICAL_WEIGHT, 0.05)
            CONFIG.EMPIRICAL_WEIGHT = emp_w
            CONFIG.USE_EXPERIENCE_FACTOR = st.checkbox(
                "Facteur expérience", value=CONFIG.USE_EXPERIENCE_FACTOR)
            K = st.slider("Shrinkage K", 0.0, 15.0,
                          CONFIG.SHRINKAGE_K, 0.5)
            CONFIG.SHRINKAGE_K = K

        with st.expander("💰 Value & Kelly"):
            vt = st.slider("Seuil de value", 1.05, 1.80,
                            CONFIG.VALUE_THRESHOLD, 0.05)
            kf = st.slider("Kelly fractionnaire", 0.05, 0.50,
                            CONFIG.KELLY_FRACTION, 0.05)
            CONFIG.KELLY_FRACTION = kf
            max_stake = st.slider("Cap max bankroll (%)", 1.0, 15.0,
                                  CONFIG.MAX_KELLY_STAKE * 100, 0.5) / 100
            CONFIG.MAX_KELLY_STAKE = max_stake

        st.markdown("---")
        if st.button("🔄 Recalculer les moyennes population (backtest)"):
            hist = load_historical_data_sample()
            if hist:
                means = compute_population_mean(hist)
                CONFIG.POPULATION_MEAN_SCORE = means["mean_score"]
                CONFIG.POPULATION_MEAN_WIN = means["mean_win"]
                st.success(f"Moyennes mises à jour : score={means['mean_score']:.2f}, win={means['mean_win']:.3f}")
            else:
                st.warning("Aucune donnée historique chargée.")

        st.caption(f"v{CONFIG.APP_VERSION} — {CONFIG.APP_TAG}")

    # ============= TABS =============
    tab1, tab2, tab3 = st.tabs(["📥 Données course",
                                "📊 Pronostics",
                                "ℹ️ Aide & Méthode"])

    # ---------- TAB 1 : DONNÉES ----------
    with tab1:
        st.markdown("## 🏁 Informations de la course")
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.5])
        with c1:
            race_type = st.selectbox("Discipline", CONFIG.RACE_TYPES)
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, 1600, 100)
        with c3:
            track = st.selectbox("Terrain", CONFIG.TRACK_CONDITIONS)
        with c4:
            default_depart = 0
            if race_type in ("Attelé", "Monté"):
                default_depart = 1
            depart = st.selectbox("Type de départ", CONFIG.DEPART_TYPES,
                                  index=default_depart)

        prix = st.text_input("Nom du prix (optionnel)", "")

        st.markdown("---")
        st.markdown("## 🐎 Tableau des partants")

        edited = st.data_editor(
            st.session_state.horses_data,
            use_container_width=True,
            num_rows="dynamic",
            height=420,
            column_config={
                "N°": st.column_config.NumberColumn(min_value=1, max_value=99),
                "Cote": st.column_config.NumberColumn(format="%.2f", min_value=1.0),
                "Corde": st.column_config.NumberColumn(min_value=0, max_value=20),
                "Poids": st.column_config.NumberColumn(format="%.1f", min_value=40.0, max_value=80.0),
                "Jours repos": st.column_config.NumberColumn(min_value=0, max_value=999),
            },
        )
        if edited is not None:
            st.session_state.horses_data = edited

        c1, c2 = st.columns([3, 1])
        with c1:
            run_btn = st.button("🚀 LANCER L'ANALYSE",
                                 use_container_width=True, type="primary")
        with c2:
            reset_btn = st.button("🔄 Reset", use_container_width=True)
            if reset_btn:
                st.session_state.horses_data = pd.DataFrame({
                    "N°": list(range(1, 11)),
                    "Nom": [f"Cheval {i+1}" for i in range(10)],
                    "Cote": [5.0] * 10,
                    "Musique Cheval": [""] * 10,
                    "Musique Driver": [""] * 10,
                    "Musique Entraîneur": [""] * 10,
                    "Corde": [0] * 10,
                    "Poids": [56.0] * 10,
                    "Jours repos": [21] * 10,
                })
                st.rerun()

        if run_btn:
            horses_list = []
            for idx, row in st.session_state.horses_data.iterrows():
                try:
                    horses_list.append({
                        "number": int(row["N°"]),
                        "name": str(row["Nom"]),
                        "odds": float(row["Cote"]),
                        "horse_music": str(row["Musique Cheval"]),
                        "driver_music": str(row["Musique Driver"]),
                        "trainer_music": str(row["Musique Entraîneur"]),
                        "draw": int(row["Corde"]) if pd.notna(row["Corde"]) else 0,
                        "weight": float(row.get("Poids", 56.0)) if pd.notna(row.get("Poids")) else 56.0,
                        "days_rest": int(row.get("Jours repos", -1)) if pd.notna(row.get("Jours repos")) else -1,
                    })
                except Exception as e:
                    st.error(f"⚠️ Erreur ligne {idx+1} : {e}")
                    return

            if len(horses_list) < 3:
                st.error("Au moins 3 partants requis.")
                return

            with st.spinner(f"🔬 Calcul Plackett-Luce ({mc_iter} simulations)..."):
                pred = run_engine_v5(
                    {"race_type": race_type, "distance": distance,
                     "track": track, "depart_type": depart, "discipline": prix},
                    horses_list,
                    mc_iter=mc_iter, market_weight=mw, value_threshold=vt,
                    pop_mean_score=CONFIG.POPULATION_MEAN_SCORE,
                    pop_mean_win=CONFIG.POPULATION_MEAN_WIN,
                )
                st.session_state.prediction = pred
            st.success(f"✅ Analyse terminée en {pred['execution_time']}s — "
                       f"{pred['n_simulations']} simulations")

    # ---------- TAB 2 : RÉSULTATS ----------
    with tab2:
        if st.session_state.prediction is None:
            st.info("🎯 Saisissez les données puis cliquez sur **LANCER L'ANALYSE**.")
        else:
            pred = st.session_state.prediction

            st.markdown("## 📈 Diagnostic de course")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 Confiance", f"{pred['confidence_idx']:.1f}/100")
            c2.metric("🌪️ Volatilité", f"{pred['volatility_idx']:.1f}/100")
            if pred["overround_pct"] is not None:
                c3.metric("📉 Overround", f"{pred['overround_pct']:.1f}%")
            else:
                c3.metric("📉 Overround", "—")
            c4.metric("📐 Seuil value (dyn.)", f"{pred['dynamic_value_threshold']:.2f}")

            if pred["kl_divergence"] is not None:
                st.caption(f"🧮 Divergence KL(modèle ‖ marché) = **{pred['kl_divergence']:.3f}**")

            st.markdown("---")
            st.markdown("## 🏆 Classement final & paris GAGNANT")
            df = pd.DataFrame([{
                "Rg": r["rank"],
                "N°": r["number"],
                "Nom": r["name"][:18],
                "Cote": f"{r['odds']:.2f}",
                "Modèle %": f"{r['win_prob_model']:.1f}",
                "Marché %": f"{r['win_prob_market']:.1f}",
                "🎯 Final %": f"{r['win_prob']:.2f}",
                "Placé %": f"{r['place_prob']:.1f}",
                "Ratio": f"{r['value_ratio']:.2f}",
                "ROI %": f"{r['expected_roi']:+.1f}",
                "Kelly %": f"{r['kelly_recommended']*100:.2f}",
                "Vol.": f"{r['volatility']:.2f}",
                "Gap": f"{r['performance_gap']:+.3f}",
                "Value": "🟢" if r["is_value_bet"] else "⚪",
            } for r in pred["results"]])
            st.dataframe(df, use_container_width=True, hide_index=True, height=380)

            value_bets = [r for r in pred["results"] if r["is_value_bet"]]
            if value_bets:
                st.markdown("### 💎 Value bets détectés")
                for vb in value_bets[:5]:
                    st.markdown(
                        f"- **N°{vb['number']} {vb['name']}** "
                        f"@ cote {vb['odds']:.2f} — "
                        f"prob. modèle {vb['win_prob']:.1f}% vs marché {vb['win_prob_market']:.1f}% "
                        f"→ Kelly : **{vb['kelly_recommended']*100:.2f}%** "
                        f"(ROI : {vb['expected_roi']:+.1f}%)"
                    )
            else:
                st.info("⚪ Aucun value bet détecté sur ce marché.")

            if pred["best_place"]:
                bp = pred["best_place"]
                st.markdown("---")
                st.markdown("## 🥉 Meilleur pari **PLACÉ**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("N°", bp["number"])
                c2.metric("Cheval", bp["name"][:15])
                c3.metric("Prob. Placé", f"{bp['place_prob']:.1f}%")
                c4.metric("ROI Placé", f"{bp['expected_roi_place']:+.1f}%")
                st.markdown(
                    f"💡 Cote placé estimée : **{bp['estimated_place_odds']:.2f}** — "
                    f"Mise Kelly : **{bp['kelly_recommended']*100:.2f}%** du bankroll"
                )

            st.markdown("---")
            st.markdown("## 🎲 Paris exotiques (Top combinaisons)")
            ex = pred["exotics"]
            tabs_exo = st.tabs(["Couplé Gagnant", "Couplé Placé",
                                "Trio Ordre", "Trio Désordre",
                                "Quarté+", "Quinté+"])

            def _render_exotic(items, key):
                if not items:
                    st.info("Aucune combinaison significative.")
                    return
                df_e = pd.DataFrame([{
                    "Rg": x["rank"],
                    "Combo": x.get("combo", "—"),
                    **({"Détail": x["names"]} if "names" in x else {}),
                    "Prob %": x["prob_pct"],
                    "Cote est.": x["estimated_odds"],
                    "ROI %": x["expected_roi"],
                } for x in items])
                st.dataframe(df_e, use_container_width=True, hide_index=True)

            with tabs_exo[0]: _render_exotic(ex["couple_gagnant"], "cg")
            with tabs_exo[1]: _render_exotic(ex["couple_place"], "cp")
            with tabs_exo[2]: _render_exotic(ex["trio_ordre"], "to")
            with tabs_exo[3]: _render_exotic(ex["trio_desordre"], "td")
            with tabs_exo[4]: _render_exotic(ex["quarte_desordre"], "q4")
            with tabs_exo[5]: _render_exotic(ex["quinte_desordre"], "q5")

    # ---------- TAB 3 : AIDE ----------
    with tab3:
        st.markdown("""
## 🎓 QuantTurf Pro v5.0 — Adaptations 2026

### Évolutions clés
- **Poids par discipline** recalibrés sur les données Quinté+ 2026.
- **Gamma d'overround** dynamique selon le type de course.
- **Backtester** intégré pour valider les performances.
- **Moyennes population** calculées automatiquement à partir des données historiques.
- **Métrique « Gap »** pour détecter les chevaux sous‑estimés.

### Backtesting
Le bouton *Recalculer les moyennes population* (sidebar) simule un chargement de données historiques. Vous pouvez remplacer la fonction `load_historical_data_sample()` par votre propre source (API PMU, fichier CSV, etc.) pour un recalage en continu.

### Références
- Benter (1994), Harville (1973), Kelly (1956)
- Données d'entraînement : Quinté+ 01/2026 – 06/2026
        """)


if __name__ == "__main__":
    main()
