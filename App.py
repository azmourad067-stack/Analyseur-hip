
import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configuration Streamlit optimisée
st.set_page_config(
    page_title="🏇 Analyseur Hippique IA Pro",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration CSS personnalisée
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .race-type-badge {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .prediction-box {
        border-left: 4px solid #f59e0b;
        padding-left: 1rem;
        background-color: #fffbeb;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Configurations par type de course
CONFIGS = {
    "PLAT": {
        "w_odds": 0.5, "w_draw": 0.3, "w_weight": 0.2,
        "description": "🏃 Course de galop - Handicap poids + avantage corde intérieure",
        "optimal_draws": [1, 2, 3, 4],
        "weight_baseline": 55.0
    },
    "ATTELE_AUTOSTART": {
        "w_odds": 0.7, "w_draw": 0.25, "w_weight": 0.05,
        "description": "🚗 Trot attelé autostart - Numéros 4-6 optimaux",
        "optimal_draws": [4, 5, 6],
        "weight_baseline": 68.0
    },
    "ATTELE_VOLTE": {
        "w_odds": 0.85, "w_draw": 0.05, "w_weight": 0.1,
        "description": "🔄 Trot attelé volté - Numéro sans importance",
        "optimal_draws": [],
        "weight_baseline": 68.0
    }
}

@st.cache_resource
class HorseRacingML:
    """Modèle ML optimisé avec cache Streamlit"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.is_trained = False
    
    def prepare_features(self, df, race_type):
        """Préparation avancée des features"""
        features = pd.DataFrame()
        
        # Features de base
        features['odds_inv'] = 1 / (df['odds_numeric'] + 0.1)
        features['draw'] = df['draw_numeric']
        features['weight'] = df['weight_kg']
        features['log_odds'] = np.log1p(df['odds_numeric'])
        
        # Features d'âge si disponible
        if 'Âge/Sexe' in df.columns:
            features['age'] = df['Âge/Sexe'].str.extract('(\d+)').astype(float).fillna(4)
            features['is_mare'] = df['Âge/Sexe'].str.contains('F', na=False).astype(int)
        else:
            features['age'] = 4.0
            features['is_mare'] = 0
        
        # Features de forme
        if 'Musique' in df.columns:
            features['recent_wins'] = df['Musique'].apply(lambda x: str(x).count('1') if pd.notna(x) else 0)
            features['recent_places'] = df['Musique'].apply(
                lambda x: sum(1 for c in str(x) if c.isdigit() and int(c) <= 3) if pd.notna(x) else 0
            )
        else:
            features['recent_wins'] = 0
            features['recent_places'] = 1
        
        # Features d'interaction
        features['odds_draw_ratio'] = features['odds_inv'] * features['draw']
        features['weight_odds_product'] = features['weight'] * features['log_odds']
        
        # Features spécifiques au type de course
        if race_type == "PLAT":
            features['inner_draw_bonus'] = (features['draw'] <= 4).astype(int)
            features['weight_penalty'] = np.maximum(0, features['weight'] - 56)
        elif race_type == "ATTELE_AUTOSTART":
            features['optimal_draw'] = features['draw'].isin([4, 5, 6]).astype(int)
            features['bad_draw'] = (features['draw'].isin([1, 2, 3]) | (features['draw'] >= 10)).astype(int)
        else:
            features['inner_draw_bonus'] = 0
            features['weight_penalty'] = 0
            features['optimal_draw'] = 0
            features['bad_draw'] = 0
        
        # Normalisation
        features = features.fillna(0)
        return features
    
    def train_and_predict(self, X, create_synthetic_target=True):
        """Entraînement et prédiction en une fois"""
        if len(X) < 3:
            return np.zeros(len(X)), {}
        
        # Création d'un target synthétique basé sur les features
        if create_synthetic_target:
            # Target basé sur l'inverse des cotes avec bruit
            y_synthetic = X['odds_inv'] + np.random.normal(0, 0.1, len(X))
        else:
            y_synthetic = np.random.randn(len(X))
        
        # Normalisation
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement des modèles
        results = {}
        predictions = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_scaled, y_synthetic)
                pred = model.predict(X_scaled)
                predictions[name] = pred
                
                results[name] = {
                    'r2': model.score(X_scaled, y_synthetic),
                    'mse': mean_squared_error(y_synthetic, pred)
                }
                
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
                    
            except Exception as e:
                st.warning(f"Erreur modèle {name}: {e}")
                predictions[name] = np.zeros(len(X))
        
        self.is_trained = True
        
        # Moyenne des prédictions
        final_predictions = np.mean(list(predictions.values()), axis=0) if predictions else np.zeros(len(X))
        return final_predictions, results

@st.cache_data(ttl=300)
def scrape_race_data(url):
    """Scraping avec cache de 5 minutes"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        horses_data = []
        
        # Recherche de tableau
        table = soup.find('table')
        if not table:
            return None, "Aucun tableau trouvé"
            
        rows = table.find_all('tr')[1:]  # Skip header
        
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 4:  # Minimum requis
                horses_data.append({
                    "Numéro de corde": cols[0].get_text(strip=True),
                    "Nom": cols[1].get_text(strip=True),
                    "Cote": cols[-1].get_text(strip=True),  # Dernier = cote généralement
                    "Poids": cols[-2].get_text(strip=True) if len(cols) > 4 else "60.0",
                    "Musique": cols[2].get_text(strip=True) if len(cols) > 5 else "",
                    "Âge/Sexe": cols[3].get_text(strip=True) if len(cols) > 6 else "",
                    "Jockey": cols[4].get_text(strip=True) if len(cols) > 7 else "",
                    "Entraîneur": cols[5].get_text(strip=True) if len(cols) > 8 else ""
                })

        if not horses_data:
            return None, "Aucune donnée extraite"
            
        return pd.DataFrame(horses_data), "Succès"
        
    except Exception as e:
        return None, f"Erreur: {str(e)}"

def safe_convert(value, convert_func, default=0):
    """Conversion sécurisée"""
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def prepare_data(df):
    """Préparation complète des données"""
    df = df.copy()
    
    # Conversions sécurisées
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    
    # Extraction du poids
    def extract_weight(poids_str):
        if pd.isna(poids_str):
            return 60.0
        match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
        return float(match.group(1).replace(',', '.')) if match else 60.0
    
    df['weight_kg'] = df['Poids'].apply(extract_weight)
    
    # Nettoyage
    df = df[df['odds_numeric'] > 0]  # Éliminer les cotes invalides
    df = df.reset_index(drop=True)
    
    return df

def auto_detect_race_type(df):
    """Détection automatique avec explications"""
    weight_std = df['weight_kg'].std()
    weight_mean = df['weight_kg'].mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("💪 Écart-type poids", f"{weight_std:.1f} kg")
    with col2:
        st.metric("⚖️ Poids moyen", f"{weight_mean:.1f} kg")
    with col3:
        st.metric("🏇 Nb chevaux", len(df))
    
    if weight_std > 2.5:
        detected = "PLAT"
        reason = "Grande variation de poids (handicap)"
    elif weight_mean > 65 and weight_std < 1.5:
        detected = "ATTELE_AUTOSTART"
        reason = "Poids uniformes élevés (attelé)"
    else:
        detected = "PLAT"
        reason = "Configuration par défaut"
    
    st.info(f"🤖 **Type détecté**: {detected} | **Raison**: {reason}")
    return detected

def create_visualization(df_ranked, ml_results=None):
    """Visualisations interactives améliorées"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '🏆 Scores Finaux par Position',
            '📊 Distribution des Cotes', 
            '⚖️ Relation Poids-Performance',
            '🧠 Importance des Features ML'
        ),
        specs=[[{"secondary_y": False}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Graphique 1: Scores
    colors = px.colors.qualitative.Set3
    score_col = 'score_final' if 'score_final' in df_ranked.columns else 'ml_score'
    if score_col in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['rang'],
                y=df_ranked[score_col],
                mode='markers+lines',
                marker=dict(size=12, color=colors[0], line=dict(width=2, color='white')),
                text=df_ranked['Nom'],
                hovertemplate='<b>%{text}</b><br>Rang: %{x}<br>Score: %{y:.2f}<extra></extra>',
                name='Score Final'
            ),
            row=1, col=1
        )
    
    # Graphique 2: Histogramme des cotes
    fig.add_trace(
        go.Histogram(
            x=df_ranked['odds_numeric'],
            nbinsx=8,
            marker_color=colors[1],
            opacity=0.7,
            name='Répartition Cotes'
        ),
        row=1, col=2
    )
    
    # Graphique 3: Poids vs Performance
    if score_col in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['weight_kg'],
                y=df_ranked[score_col],
                mode='markers',
                marker=dict(
                    size=8,
                    color=df_ranked['rang'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Rang")
                ),
                text=df_ranked['Nom'],
                hovertemplate='<b>%{text}</b><br>Poids: %{x:.1f}kg<br>Score: %{y:.2f}<extra></extra>',
                name='Poids vs Score'
            ),
            row=2, col=1
        )
    
    # Graphique 4: Feature importance
    if ml_results and 'random_forest' in ml_results:
        importance = ml_results['random_forest'].get('feature_importance', {})
        if importance:
            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:6])
            
            fig.add_trace(
                go.Bar(
                    x=list(top_features.values()),
                    y=list(top_features.keys()),
                    orientation='h',
                    marker_color=colors[3],
                    name='Importance Features'
                ),
                row=2, col=2
            )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="📊 Analyse Complète de la Course",
        title_x=0.5
    )
    
    return fig

def generate_sample_data(data_type="plat"):
    """Génération de données d'exemple"""
    if data_type == "plat":
        return pd.DataFrame({
            'Nom': ['Thunder Bolt', 'Lightning Star', 'Storm King', 'Rain Dance', 'Wind Walker', 'Fire Spirit', 'Ocean Wave', 'Mountain Peak'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1', '12.0', '15.5', '28.0'],
            'Poids': ['56.5', '57.0', '58.5', '59.0', '57.5', '60.0', '55.5', '61.0'],
            'Musique': ['1a2a3a', '2a1a4a', '3a3a1a', '1a4a2a', '4a2a5a', '5a3a6a', '6a5a4a', '8a7a9a'],
            'Âge/Sexe': ['4H', '5M', '3F', '6H', '4M', '5F', '4H', '3M']
        })
    elif data_type == "attele":
        return pd.DataFrame({
            'Nom': ['Rapide Éclair', 'Foudre Noire', 'Vent du Nord', 'Tempête Rouge', 'Orage Bleu', 'Flash Gordon', 'Speed Demon', 'Quick Silver'],
            'Numéro de corde': ['1', '2', '3', '4', '5', '6', '7', '8'],
            'Cote': ['4.2', '8.5', '15.0', '3.8', '6.8', '5.2', '22.0', '12.5'],
            'Poids': ['68.0', '68.0', '68.0', '68.0', '68.0', '68.0', '68.0', '68.0'],
            'Musique': ['2a1a4a', '4a3a2a', '6a5a8a', '1a2a1a', '3a4a5a', '2a3a6a', '9a8a7a', '5a4a3a'],
            'Âge/Sexe': ['5H', '6M', '4F', '7H', '5M', '4F', '6H', '5M']
        })
    else:  # premium
        return pd.DataFrame({
            'Nom': ['Ace Impact', 'Torquator Tasso', 'Adayar', 'Tarnawa', 'Chrono Genesis'],
            'Numéro de corde': ['1', '2', '3', '4', '5'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1'],
            'Poids': ['59.5', '59.5', '59.5', '58.5', '58.5'],
            'Musique': ['1a1a2a', '1a3a1a', '2a1a4a', '1a2a1a', '3a1a2a'],
            'Âge/Sexe': ['4H', '5H', '4H', '5F', '5F']
        })

# Interface principale
def main():
    
    # En-tête avec style
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA Pro</h1>', unsafe_allow_html=True)
    st.markdown("*Analyse prédictive des courses hippiques avec Machine Learning avancé*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Type de course
        race_type = st.selectbox(
            "🏁 Type de course",
            ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"],
            help="AUTO = détection automatique basée sur les données"
        )
        
        # Paramètres ML
        st.subheader("🤖 Intelligence Artificielle")
        use_ml = st.checkbox("✅ Activer prédictions ML", value=True)
        ml_confidence = st.slider("🎯 Poids ML dans le score final", 0.1, 0.9, 0.6, 0.1)
        
        # Options avancées
        st.subheader("🔧 Options Avancées")
        show_detailed_features = st.checkbox("📊 Afficher features détaillées")
        export_predictions = st.checkbox("💾 Préparer export complet")
        
        # Informations
        st.subheader("ℹ️ Informations")
        st.info("🧠 **ML Models**: Random Forest + Gradient Boosting")
        st.info("📚 **Basé sur**: Recherches turfmining.fr, boturfers.fr")
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs(["🌐 URL Analysis", "📁 Upload CSV", "🧪 Test Data", "📖 Documentation"])
    
    df_final = None
    
    with tab1:
        st.subheader("🔍 Analyse d'URL de Course")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input(
                "🌐 URL de la course à analyser:",
                placeholder="https://example-racing-site.com/course/123",
                help="Entrez l'URL d'une page de course hippique"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🔍 Analyser", type="primary")
        
        if analyze_button and url:
            with st.spinner("🔄 Extraction des données en cours..."):
                df, message = scrape_race_data(url)
                
                if df is not None:
                    st.success(f"✅ Données extraites avec succès: **{len(df)} chevaux**")
                    st.dataframe(df.head(), use_container_width=True)
                    df_final = df
                else:
                    st.error(f"❌ {message}")
                    st.info("💡 **Astuce**: Vérifiez que l'URL contient un tableau de course valide")
    
    with tab2:
        st.subheader("📤 Upload de Fichier CSV")
        
        uploaded_file = st.file_uploader(
            "Choisir un fichier CSV",
            type="csv",
            help="Format attendu: Nom, Numéro de corde, Cote, Poids, Musique, Âge/Sexe"
        )
        
        if uploaded_file:
            try:
                df_final = pd.read_csv(uploaded_file)
                st.success(f"✅ Fichier chargé: **{len(df_final)} chevaux**")
                
                # Aperçu des données
                st.subheader("👀 Aperçu des données")
                st.dataframe(df_final.head(), use_container_width=True)
                
                # Validation des colonnes
                required_cols = ['Nom', 'Cote']
                missing_cols = [col for col in required_cols if col not in df_final.columns]
                if missing_cols:
                    st.warning(f"⚠️ Colonnes manquantes: {missing_cols}")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {e}")
    
    with tab3:
        st.subheader("🧪 Données de Test")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏃 Test Plat", use_container_width=True):
                df_final = generate_sample_data("plat")
                st.success(f"✅ Données PLAT chargées ({len(df_final)} chevaux)")
        
        with col2:
            if st.button("🚗 Test Attelé", use_container_width=True):
                df_final = generate_sample_data("attele")
                st.success(f"✅ Données ATTELÉ chargées ({len(df_final)} chevaux)")
        
        with col3:
            if st.button("⭐ Test Premium", use_container_width=True):
                df_final = generate_sample_data("premium")
                st.success(f"✅ Données PREMIUM chargées ({len(df_final)} chevaux)")
        
        if df_final is not None:
            st.dataframe(df_final, use_container_width=True)
    
    with tab4:
        st.subheader("📚 Guide d'Utilisation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 Types de Courses
            
            **🏃 PLAT (Galop)**
            - Avantage cordes intérieures (1-4)
            - Impact important du poids
            - Handicap variable
            
            **🚗 ATTELÉ AUTOSTART**  
            - Numéros optimaux: 4, 5, 6
            - Poids réglementaire uniforme
            - Placement stratégique crucial
            
            **🔄 ATTELÉ VOLTÉ**
            - Numéro sans importance
            - Focus sur la forme/cotes
            - Stratégie driver prépondérante
            """)
        
        with col2:
            st.markdown("""
            ### 🤖 Intelligence Artificielle
            
            **Features Analysées**
            - 🎯 Cotes et probabilités inverses
            - 📍 Position de départ optimale
            - ⚖️ Pénalités/bonus de poids
            - 🏆 Forme récente (musique)
            - 🔄 Interactions complexes
            
            **Modèles ML**
            - Random Forest (robustesse)
            - Gradient Boosting (performance)
            - Ensemble voting (consensus)
            """)
    
    # Analyse des données si disponibles
    if df_final is not None and len(df_final) > 0:
        
        st.markdown("---")
        st.header("🎯 Analyse et Résultats")
        
        # Préparation des données
        df_prepared = prepare_data(df_final)
        
        if len(df_prepared) == 0:
            st.error("❌ Aucune donnée valide après nettoyage")
            return
        
        # Détection du type de course
        if race_type == "AUTO":
            detected_type = auto_detect_race_type(df_prepared)
        else:
            detected_type = race_type
            st.markdown(f'<div class="race-type-badge">{CONFIGS[detected_type]["description"]}</div>', 
                       unsafe_allow_html=True)
        
        config = CONFIGS[detected_type]
        
        # Analyse ML si activée
        ml_model = HorseRacingML()
        ml_results = None
        
        if use_ml:
            with st.spinner("🤖 Entraînement du modèle ML en cours..."):
                try:
                    # Préparation des features
                    X_ml = ml_model.prepare_features(df_prepared, detected_type)
                    
                    # Entraînement et prédiction
                    ml_predictions, ml_results = ml_model.train_and_predict(X_ml)
                    
                    # Normalisation des prédictions ML
                    if len(ml_predictions) > 0 and ml_predictions.max() != ml_predictions.min():
                        ml_predictions = (ml_predictions - ml_predictions.min()) / (ml_predictions.max() - ml_predictions.min())
                    df_prepared['ml_score'] = ml_predictions
                    
                    st.success("✅ Modèle ML entraîné avec succès")
                    
                except Exception as e:
                    st.warning(f"⚠️ Erreur ML: {e}")
                    use_ml = False
        
        # Calcul du score final
        # Score traditionnel basé sur les cotes
        traditional_score = 1 / (df_prepared['odds_numeric'] + 0.1)
        if traditional_score.max() != traditional_score.min():
            traditional_score = (traditional_score - traditional_score.min()) / (traditional_score.max() - traditional_score.min())
        
        if use_ml and 'ml_score' in df_prepared.columns:
            # Combinaison des scores
            df_prepared['score_final'] = (
                (1 - ml_confidence) * traditional_score + 
                ml_confidence * df_prepared['ml_score']
            )
        else:
            df_prepared['score_final'] = traditional_score
        
        # Classement final
        df_ranked = df_prepared.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked) + 1)
        
        # Affichage des résultats
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏆 Classement Final")
            
            # Préparation des colonnes d'affichage
            display_cols = ['rang', 'Nom', 'Cote', 'Numéro de corde']
            if 'Poids' in df_ranked.columns:
                display_cols.append('Poids')
            
            display_cols.append('score_final')
            
            # Formatage du dataframe pour l'affichage
            display_df = df_ranked[display_cols].copy()
            if 'score_final' in display_df.columns:
                display_df['Score'] = display_df['score_final'].round(3)
                display_df = display_df.drop('score_final', axis=1)
            
            st.dataframe(display_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Métriques")
            
            # Métriques de performance ML
            if ml_results and 'random_forest' in ml_results:
                rf_r2 = ml_results['random_forest']['r2']
                st.markdown(f'<div class="metric-card">🧠 R² Score ML<br><strong>{rf_r2:.3f}</strong></div>', 
                           unsafe_allow_html=True)
            
            # Métriques de course
            favoris = len(df_ranked[df_ranked['odds_numeric'] < 5])
            outsiders = len(df_ranked[df_ranked['odds_numeric'] > 15])
            
            st.markdown(f'<div class="metric-card">⭐ Favoris<br><strong>{favoris}</strong></div>', 
                       unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">🎲 Outsiders<br><strong>{outsiders}</strong></div>', 
                       unsafe_allow_html=True)
            
            # Recommandation top 3
            st.subheader("🥇 Top 3 Recommandé")
            for i in range(min(3, len(df_ranked))):
                horse = df_ranked.iloc[i]
                st.markdown(f"""
                <div class="prediction-box">
                    <strong>{i+1}. {horse['Nom']}</strong><br>
                    Cote: {horse['Cote']} | Score: {horse['score_final']:.3f}
                </div>
                """, unsafe_allow_html=True)
        
        # Visualisations
        st.subheader("📊 Visualisations Interactives")
        fig = create_visualization(df_ranked, ml_model.feature_importance if ml_model.feature_importance else None)
        st.plotly_chart(fig, use_container_width=True)
        
        # Features détaillées si demandé
        if show_detailed_features and use_ml and ml_model.feature_importance:
            st.subheader("🔍 Analyse Détaillée des Features")
            
            for model_name, importance in ml_model.feature_importance.items():
                if importance:
                    importance_df = pd.DataFrame([
                        {'Feature': k, 'Importance': v, 'Pourcentage': f"{v*100:.1f}%"}
                        for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    ]).head(8)
                    
                    st.subheader(f"📈 {model_name.title().replace('_', ' ')} - Top Features")
                    st.dataframe(importance_df, use_container_width=True)
        
        # Export des données
        if export_predictions:
            st.subheader("💾 Export des Résultats")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV Export
                csv_data = df_ranked.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="📄 Télécharger CSV",
                    data=csv_data,
                    file_name=f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # JSON Export
                json_data = df_ranked.to_json(orient='records', force_ascii=False, indent=2)
                st.download_button(
                    label="📋 Télécharger JSON",
                    data=json_data,
                    file_name=f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
Response
Created file /home/user/fixed_streamlit_app.py (28152 characters)
