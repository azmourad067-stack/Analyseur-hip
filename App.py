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
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🏇 Analyseur Hippique IA",
    page_icon="🏇",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .prediction-box {
        border-left: 4px solid #f59e0b;
        padding-left: 1rem;
        background-color: #fffbeb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

CONFIGS = {
    "PLAT": {
        "description": "🏃 Course de galop - Handicap poids + avantage corde intérieure",
        "optimal_draws": [1, 2, 3, 4]
    },
    "ATTELE_AUTOSTART": {
        "description": "🚗 Trot attelé autostart - Numéros 4-6 optimaux", 
        "optimal_draws": [4, 5, 6]
    },
    "ATTELE_VOLTE": {
        "description": "🔄 Trot attelé volté - Numéro sans importance",
        "optimal_draws": []
    }
}

@st.cache_resource
class HorseRacingML:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.is_trained = False
    
    def prepare_features(self, df):
        features = pd.DataFrame()
        features['odds_inv'] = 1 / (df['odds_numeric'] + 0.1)
        features['draw'] = df['draw_numeric']
        features['weight'] = df['weight_kg']
        features['log_odds'] = np.log1p(df['odds_numeric'])
        
        if 'Âge/Sexe' in df.columns:
            features['age'] = df['Âge/Sexe'].str.extract('(\d+)').astype(float).fillna(4)
            features['is_mare'] = df['Âge/Sexe'].str.contains('F', na=False).astype(int)
        else:
            features['age'] = 4.0
            features['is_mare'] = 0
        
        if 'Musique' in df.columns:
            features['recent_wins'] = df['Musique'].apply(lambda x: str(x).count('1') if pd.notna(x) else 0)
            features['recent_places'] = df['Musique'].apply(
                lambda x: sum(1 for c in str(x) if c.isdigit() and int(c) <= 3) if pd.notna(x) else 0
            )
        else:
            features['recent_wins'] = 0
            features['recent_places'] = 1
        
        features['odds_draw_ratio'] = features['odds_inv'] * features['draw']
        features['weight_odds_product'] = features['weight'] * features['log_odds']
        
        return features.fillna(0)
    
    def train_and_predict(self, X):
        if len(X) < 3:
            return np.zeros(len(X)), {}
        
        y_synthetic = X['odds_inv'] + np.random.normal(0, 0.1, len(X))
        X_scaled = self.scaler.fit_transform(X)
        
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
        final_predictions = np.mean(list(predictions.values()), axis=0) if predictions else np.zeros(len(X))
        return final_predictions, results

@st.cache_data(ttl=300)
def scrape_race_data(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return None, f"Erreur HTTP {response.status_code}"

        soup = BeautifulSoup(response.content, 'html.parser')
        horses_data = []
        
        table = soup.find('table')
        if not table:
            return None, "Aucun tableau trouvé"
            
        rows = table.find_all('tr')[1:]
        
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 4:
                horses_data.append({
                    "Numéro de corde": cols[0].get_text(strip=True),
                    "Nom": cols[1].get_text(strip=True),
                    "Cote": cols[-1].get_text(strip=True),
                    "Poids": cols[-2].get_text(strip=True) if len(cols) > 4 else "60.0",
                    "Musique": cols[2].get_text(strip=True) if len(cols) > 5 else "",
                    "Âge/Sexe": cols[3].get_text(strip=True) if len(cols) > 6 else "",
                })

        if not horses_data:
            return None, "Aucune donnée extraite"
            
        return pd.DataFrame(horses_data), "Succès"
        
    except Exception as e:
        return None, f"Erreur: {str(e)}"

def safe_convert(value, convert_func, default=0):
    try:
        if pd.isna(value):
            return default
        cleaned = str(value).replace(',', '.').strip()
        return convert_func(cleaned)
    except:
        return default

def prepare_data(df):
    df = df.copy()
    df['odds_numeric'] = df['Cote'].apply(lambda x: safe_convert(x, float, 999))
    df['draw_numeric'] = df['Numéro de corde'].apply(lambda x: safe_convert(x, int, 1))
    
    def extract_weight(poids_str):
        if pd.isna(poids_str):
            return 60.0
        match = re.search(r'(\d+(?:[.,]\d+)?)', str(poids_str))
        return float(match.group(1).replace(',', '.')) if match else 60.0
    
    df['weight_kg'] = df['Poids'].apply(extract_weight)
    df = df[df['odds_numeric'] > 0]
    df = df.reset_index(drop=True)
    return df

def auto_detect_race_type(df):
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
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('🏆 Scores par Position', '📊 Distribution Cotes', '⚖️ Poids vs Performance', '🧠 Features ML'),
        specs=[[{"secondary_y": False}, {"type": "histogram"}], [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    colors = px.colors.qualitative.Set3
    score_col = 'score_final' if 'score_final' in df_ranked.columns else 'ml_score'
    
    if score_col in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['rang'], y=df_ranked[score_col],
                mode='markers+lines', marker=dict(size=12, color=colors[0]),
                text=df_ranked['Nom'], name='Score Final'
            ), row=1, col=1
        )
    
    fig.add_trace(
        go.Histogram(x=df_ranked['odds_numeric'], nbinsx=8, marker_color=colors[1], name='Cotes'),
        row=1, col=2
    )
    
    if score_col in df_ranked.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ranked['weight_kg'], y=df_ranked[score_col],
                mode='markers', marker=dict(size=8, color=df_ranked['rang'], colorscale='Viridis'),
                text=df_ranked['Nom'], name='Poids vs Score'
            ), row=2, col=1
        )
    
    if ml_results and 'random_forest' in ml_results:
        importance = ml_results['random_forest'].get('feature_importance', {})
        if importance:
            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:6])
            fig.add_trace(
                go.Bar(x=list(top_features.values()), y=list(top_features.keys()), 
                       orientation='h', marker_color=colors[3], name='Importance'),
                row=2, col=2
            )
    
    fig.update_layout(height=600, showlegend=True, title_text="📊 Analyse Complète", title_x=0.5)
    return fig

def generate_sample_data(data_type="plat"):
    if data_type == "plat":
        return pd.DataFrame({
            'Nom': ['Thunder Bolt', 'Lightning Star', 'Storm King', 'Rain Dance', 'Wind Walker'],
            'Numéro de corde': ['1', '2', '3', '4', '5'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1'],
            'Poids': ['56.5', '57.0', '58.5', '59.0', '57.5'],
            'Musique': ['1a2a3a', '2a1a4a', '3a3a1a', '1a4a2a', '4a2a5a'],
            'Âge/Sexe': ['4H', '5M', '3F', '6H', '4M']
        })
    elif data_type == "attele":
        return pd.DataFrame({
            'Nom': ['Rapide Éclair', 'Foudre Noire', 'Vent du Nord', 'Tempête Rouge', 'Orage Bleu'],
            'Numéro de corde': ['1', '2', '3', '4', '5'],
            'Cote': ['4.2', '8.5', '15.0', '3.8', '6.8'],
            'Poids': ['68.0', '68.0', '68.0', '68.0', '68.0'],
            'Musique': ['2a1a4a', '4a3a2a', '6a5a8a', '1a2a1a', '3a4a5a'],
            'Âge/Sexe': ['5H', '6M', '4F', '7H', '5M']
        })
    else:
        return pd.DataFrame({
            'Nom': ['Ace Impact', 'Torquator Tasso', 'Adayar', 'Tarnawa', 'Chrono Genesis'],
            'Numéro de corde': ['1', '2', '3', '4', '5'],
            'Cote': ['3.2', '4.8', '7.5', '6.2', '9.1'],
            'Poids': ['59.5', '59.5', '59.5', '58.5', '58.5'],
            'Musique': ['1a1a2a', '1a3a1a', '2a1a4a', '1a2a1a', '3a1a2a'],
            'Âge/Sexe': ['4H', '5H', '4H', '5F', '5F']
        })

def main():
    st.markdown('<h1 class="main-header">🏇 Analyseur Hippique IA</h1>', unsafe_allow_html=True)
    st.markdown("*Analyse prédictive des courses hippiques avec Machine Learning*")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        race_type = st.selectbox("🏁 Type de course", ["AUTO", "PLAT", "ATTELE_AUTOSTART", "ATTELE_VOLTE"])
        use_ml = st.checkbox("✅ Activer prédictions ML", value=True)
        ml_confidence = st.slider("🎯 Poids ML", 0.1, 0.9, 0.6, 0.1)
        
        st.subheader("ℹ️ Informations")
        st.info("🧠 **ML**: Random Forest + Gradient Boosting")
        st.info("📚 **Sources**: turfmining.fr, boturfers.fr")
    
    tab1, tab2, tab3 = st.tabs(["🌐 URL Analysis", "📁 Upload CSV", "🧪 Test Data"])
    
    df_final = None
    
    with tab1:
        st.subheader("🔍 Analyse d'URL de Course")
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input("🌐 URL de la course:", placeholder="https://example-racing-site.com/course/123")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze_button = st.button("🔍 Analyser", type="primary")
        
        if analyze_button and url:
            with st.spinner("🔄 Extraction..."):
                df, message = scrape_race_data(url)
                if df is not None:
                    st.success(f"✅ {len(df)} chevaux extraits")
                    st.dataframe(df.head())
                    df_final = df
                else:
                    st.error(f"❌ {message}")
    
    with tab2:
        st.subheader("📤 Upload CSV")
        uploaded_file = st.file_uploader("Fichier CSV", type="csv")
        if uploaded_file:
            try:
                df_final = pd.read_csv(uploaded_file)
                st.success(f"✅ {len(df_final)} chevaux chargés")
                st.dataframe(df_final.head())
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
    
    with tab3:
        st.subheader("🧪 Données de Test")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏃 Test Plat"):
                df_final = generate_sample_data("plat")
                st.success("✅ Données PLAT chargées")
        with col2:
            if st.button("🚗 Test Attelé"):
                df_final = generate_sample_data("attele")
                st.success("✅ Données ATTELÉ chargées")
        with col3:
            if st.button("⭐ Test Premium"):
                df_final = generate_sample_data("premium")
                st.success("✅ Données PREMIUM chargées")
        
        if df_final is not None:
            st.dataframe(df_final)
    
    if df_final is not None and len(df_final) > 0:
        st.markdown("---")
        st.header("🎯 Analyse et Résultats")
        
        df_prepared = prepare_data(df_final)
        if len(df_prepared) == 0:
            st.error("❌ Aucune donnée valide")
            return
        
        if race_type == "AUTO":
            detected_type = auto_detect_race_type(df_prepared)
        else:
            detected_type = race_type
            st.info(f"📋 {CONFIGS[detected_type]['description']}")
        
        ml_model = HorseRacingML()
        ml_results = None
        
        if use_ml:
            with st.spinner("🤖 ML en cours..."):
                try:
                    X_ml = ml_model.prepare_features(df_prepared)
                    ml_predictions, ml_results = ml_model.train_and_predict(X_ml)
                    
                    if len(ml_predictions) > 0 and ml_predictions.max() != ml_predictions.min():
                        ml_predictions = (ml_predictions - ml_predictions.min()) / (ml_predictions.max() - ml_predictions.min())
                    df_prepared['ml_score'] = ml_predictions
                    st.success("✅ ML entraîné")
                except Exception as e:
                    st.warning(f"⚠️ Erreur ML: {e}")
                    use_ml = False
        
        traditional_score = 1 / (df_prepared['odds_numeric'] + 0.1)
        if traditional_score.max() != traditional_score.min():
            traditional_score = (traditional_score - traditional_score.min()) / (traditional_score.max() - traditional_score.min())
        
        if use_ml and 'ml_score' in df_prepared.columns:
            df_prepared['score_final'] = (1 - ml_confidence) * traditional_score + ml_confidence * df_prepared['ml_score']
        else:
            df_prepared['score_final'] = traditional_score
        
        df_ranked = df_prepared.sort_values('score_final', ascending=False).reset_index(drop=True)
        df_ranked['rang'] = range(1, len(df_ranked) + 1)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🏆 Classement Final")
            display_cols = ['rang', 'Nom', 'Cote', 'Numéro de corde']
            if 'Poids' in df_ranked.columns:
                display_cols.append('Poids')
            display_cols.append('score_final')
            
            display_df = df_ranked[display_cols].copy()
            if 'score_final' in display_df.columns:
                display_df['Score'] = display_df['score_final'].round(3)
                display_df = display_df.drop('score_final', axis=1)
            st.dataframe(display_df, use_container_width=True)
        
        with col2:
            st.subheader("📊 Métriques")
            if ml_results and 'random_forest' in ml_results:
                rf_r2 = ml_results['random_forest']['r2']
                st.markdown(f'<div class="metric-card">🧠 R² ML<br><strong>{rf_r2:.3f}</strong></div>', unsafe_allow_html=True)
            
            favoris = len(df_ranked[df_ranked['odds_numeric'] < 5])
            outsiders = len(df_ranked[df_ranked['odds_numeric'] > 15])
            
            st.markdown(f'<div class="metric-card">⭐ Favoris<br><strong>{favoris}</strong></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card">🎲 Outsiders<br><strong>{outsiders}</strong></div>', unsafe_allow_html=True)
            
            st.subheader("🥇 Top 3")
            for i in range(min(3, len(df_ranked))):
                horse = df_ranked.iloc[i]
                st.markdown(f"""
                <div class="prediction-box">
                    <strong>{i+1}. {horse['Nom']}</strong><br>
                    Cote: {horse['Cote']} | Score: {horse['score_final']:.3f}
                </div>
                """, unsafe_allow_html=True)
        
        st.subheader("📊 Visualisations")
        fig = create_visualization(df_ranked, ml_model.feature_importance if ml_model.feature_importance else None)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("💾 Export")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = df_ranked.to_csv(index=False)
            st.download_button("📄 CSV", csv_data, f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with col2:
            json_data = df_ranked.to_json(orient='records', indent=2)
            st.download_button("📋 JSON", json_data, f"pronostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

if __name__ == "__main__":
    main()
Response
Created file /home/user/app_clean.py (18090 characters)
