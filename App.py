import streamlit as st
import easyocr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import re
import time
import itertools

# ------------------------------------------------------------
# Configuration de la page
# ------------------------------------------------------------
st.set_page_config(page_title="Pronostics Hippiques", layout="wide")

# ------------------------------------------------------------
# Initialisation du lecteur OCR (mise en cache)
# ------------------------------------------------------------
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['fr'], gpu=False)

# ------------------------------------------------------------
# Fonctions OCR et extraction de tableaux
# ------------------------------------------------------------
def extract_text_and_boxes(image):
    """Extrait le texte et les bounding boxes d'une image PIL."""
    reader = get_ocr_reader()
    img_np = np.array(image)
    results = reader.readtext(img_np)
    text_boxes = []
    for (bbox, text, conf) in results:
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        text_boxes.append({
            'text': text.strip(),
            'x': (x_min + x_max) / 2,
            'y': (y_min + y_max) / 2,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max
        })
    return text_boxes

def detect_table_structure(text_boxes, headers_keywords):
    """
    Détecte les lignes et colonnes d'un tableau à partir des bounding boxes.
    headers_keywords: liste de mots-clés possibles pour l'en-tête.
    Retourne un DataFrame avec les colonnes détectées.
    """
    if not text_boxes:
        return pd.DataFrame()
    
    text_boxes.sort(key=lambda x: x['y'])
    
    # Regrouper par lignes
    heights = [box['y_max'] - box['y_min'] for box in text_boxes]
    avg_height = np.mean(heights) if heights else 20
    line_threshold = avg_height * 0.8
    
    lines = []
    current_line = []
    current_y = None
    for box in text_boxes:
        if current_y is None or abs(box['y'] - current_y) < line_threshold:
            current_line.append(box)
            current_y = box['y']
        else:
            lines.append(sorted(current_line, key=lambda x: x['x']))
            current_line = [box]
            current_y = box['y']
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x['x']))
    
    # Chercher la ligne d'en-tête
    header_line = None
    header_index = 0
    for i, line in enumerate(lines):
        texts = [item['text'] for item in line]
        if any(any(kw in t for kw in headers_keywords) for t in texts):
            header_line = line
            header_index = i
            break
    
    if header_line is None:
        header_line = lines[0]
        header_index = 0
    
    columns = [item['text'] for item in header_line]
    data_lines = lines[header_index+1:]
    
    data_rows = []
    for line in data_lines:
        row = {}
        for item in line:
            best_col = None
            best_dist = float('inf')
            for col_item in header_line:
                dist = abs(item['x'] - col_item['x'])
                if dist < best_dist:
                    best_dist = dist
                    best_col = col_item['text']
            if best_col:
                row[best_col] = item['text']
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows)
    if not df.empty:
        # Réordonner selon l'ordre des colonnes de l'en-tête
        cols_present = [c for c in columns if c in df.columns]
        df = df[cols_present]
    return df

def extract_table_from_image(image, headers_keywords):
    """Fonction principale pour extraire un tableau d'une image."""
    boxes = extract_text_and_boxes(image)
    df = detect_table_structure(boxes, headers_keywords)
    return df

# ------------------------------------------------------------
# Fonctions de nettoyage et transformation des données
# ------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace('|', '1').replace('O', '0').replace('l', '1')
    text = text.strip()
    return text

def parse_percentage(pct_str):
    if isinstance(pct_str, str):
        match = re.search(r'(\d+(?:[.,]\d+)?)\s*%', pct_str)
        if match:
            return float(match.group(1).replace(',', '.')) / 100
    return None

def parse_gains(gains_str):
    if isinstance(gains_str, str):
        gains_str = gains_str.replace(' ', '').replace('\u202f', '')
        match = re.search(r'(\d+)', gains_str)
        if match:
            return int(match.group(1))
    return None

def parse_record(record_str):
    if isinstance(record_str, str):
        match = re.search(r"(\d+)'(\d+)\"?(\d?)", record_str)
        if match:
            minutes = int(match.group(1))
            secondes = int(match.group(2))
            dixiemes = int(match.group(3)) if match.group(3) else 0
            return minutes * 60 + secondes + dixiemes / 10
    return None

def parse_musique(musique_str):
    if not isinstance(musique_str, str):
        return []
    musique_str = re.sub(r'\(\d+\)', '', musique_str)
    pattern = r'(\d*[aAmM]?[aA]?|Da|Dm|0a)'
    parts = re.findall(pattern, musique_str)
    return [p for p in parts if p]

def score_musique(musique_list, max_items=5):
    """
    Calcule un score à partir des dernières performances.
    Pondération: 1a=10, 2a=8, 3a=6, 4a=5, 5a=4, 6a=3, 7a=2, 8a=1, 9a=0, Da/Dm/0a=0.
    """
    weights = {
        '1a': 10, '1m': 10,
        '2a': 8, '2m': 8,
        '3a': 6, '3m': 6,
        '4a': 5, '4m': 5,
        '5a': 4, '5m': 4,
        '6a': 3, '6m': 3,
        '7a': 2, '7m': 2,
        '8a': 1, '8m': 1,
        '9a': 0, '9m': 0,
        'Da': 0, 'Dm': 0, '0a': 0
    }
    recent = musique_list[:max_items]
    if not recent:
        return 0
    total = 0
    for perf in recent:
        perf_lower = perf.lower()
        if perf_lower in weights:
            total += weights[perf_lower]
        else:
            match = re.match(r'(\d+)a', perf_lower)
            if match:
                place = int(match.group(1))
                if 1 <= place <= 9:
                    total += max(0, 10 - place)
                else:
                    total += 0
            else:
                total += 0
    return total / len(recent)

def clean_dataframe(df, table_type):
    """Nettoie un DataFrame selon le type de tableau."""
    df = df.copy()
    df = df.dropna(how='all')
    
    for col in df.columns:
        df[col] = df[col].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    
    if table_type == 'partants':
        if 'N°' in df.columns:
            df['N°'] = pd.to_numeric(df['N°'], errors='coerce')
        if 'Gains' in df.columns:
            df['Gains'] = df['Gains'].apply(parse_gains)
        if 'Musique' in df.columns:
            df['Musique_cheval'] = df['Musique'].apply(parse_musique)
    
    elif table_type == 'drivers':
        if 'Réussite' in df.columns:
            df['Reussite_driver'] = df['Réussite'].apply(parse_percentage)
        if 'Courses' in df.columns:
            df['Courses_driver'] = pd.to_numeric(df['Courses'], errors='coerce')
        if 'Victoires' in df.columns:
            df['Victoires_driver'] = pd.to_numeric(df['Victoires'], errors='coerce')
        if 'Ecart' in df.columns:
            df['Ecart_driver'] = pd.to_numeric(df['Ecart'], errors='coerce')
        if 'Musique Driver' in df.columns:
            df['Musique_driver'] = df['Musique Driver'].apply(parse_musique)
    
    elif table_type == 'entraineurs':
        if 'Réussite' in df.columns:
            df['Reussite_entraineur'] = df['Réussite'].apply(parse_percentage)
        if 'Courses' in df.columns:
            df['Courses_entraineur'] = pd.to_numeric(df['Courses'], errors='coerce')
        if 'Victoires' in df.columns:
            df['Victoires_entraineur'] = pd.to_numeric(df['Victoires'], errors='coerce')
        if 'Ecart' in df.columns:
            df['Ecart_entraineur'] = pd.to_numeric(df['Ecart'], errors='coerce')
        if 'Musique Entraîneur' in df.columns:
            df['Musique_entraineur'] = df['Musique Entraîneur'].apply(parse_musique)
    
    elif table_type == 'records':
        if 'Record' in df.columns:
            df['Record_secondes'] = df['Record'].apply(parse_record)
    
    return df

def fusionner_donnees(partants_df, drivers_df, entraineurs_df, records_df):
    """Fusionne les différentes tables en une seule basée sur le numéro et le nom."""
    if partants_df is None or partants_df.empty:
        return pd.DataFrame()
    
    def normalize_name(name):
        if isinstance(name, str):
            name = name.strip().lower()
            name = name.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('à', 'a').replace('ç', 'c')
            return name
        return ''
    
    partants_df['cheval_norm'] = partants_df['Cheval'].apply(normalize_name)
    
    def merge_table(base_df, other_df, suffix):
        if other_df is None or other_df.empty:
            return base_df
        other_df['cheval_norm'] = other_df['Cheval'].apply(normalize_name)
        merged = pd.merge(base_df, other_df, on=['N°', 'cheval_norm'], how='left', suffixes=('', suffix))
        return merged
    
    if drivers_df is not None:
        partants_df = merge_table(partants_df, drivers_df, '_driver')
    if entraineurs_df is not None:
        partants_df = merge_table(partants_df, entraineurs_df, '_entraineur')
    if records_df is not None:
        partants_df = merge_table(partants_df, records_df, '_record')
    
    partants_df['score_musique_cheval'] = partants_df['Musique_cheval'].apply(lambda x: score_musique(x) if isinstance(x, list) else 0)
    if 'Musique_driver' in partants_df.columns:
        partants_df['score_musique_driver'] = partants_df['Musique_driver'].apply(lambda x: score_musique(x) if isinstance(x, list) else 0)
    if 'Musique_entraineur' in partants_df.columns:
        partants_df['score_musique_entraineur'] = partants_df['Musique_entraineur'].apply(lambda x: score_musique(x) if isinstance(x, list) else 0)
    
    return partants_df

# ------------------------------------------------------------
# Fonctions de scoring
# ------------------------------------------------------------
def calculer_score_cheval(row, weights=None):
    if weights is None:
        weights = {
            'record': 0.20,
            'reussite_driver': 0.15,
            'reussite_entraineur': 0.15,
            'musique_cheval': 0.20,
            'musique_driver': 0.10,
            'musique_entraineur': 0.10,
            'gains': 0.05,
            'ecart_driver': 0.03,
            'ecart_entraineur': 0.02
        }
    
    score = 0
    
    if pd.notna(row.get('Record_secondes')):
        score += weights['record'] * (1 / row['Record_secondes'])
    
    if pd.notna(row.get('Reussite_driver')):
        score += weights['reussite_driver'] * row['Reussite_driver']
    
    if pd.notna(row.get('Reussite_entraineur')):
        score += weights['reussite_entraineur'] * row['Reussite_entraineur']
    
    if pd.notna(row.get('score_musique_cheval')):
        score += weights['musique_cheval'] * (row['score_musique_cheval'] / 10)
    
    if pd.notna(row.get('score_musique_driver')):
        score += weights['musique_driver'] * (row['score_musique_driver'] / 10)
    
    if pd.notna(row.get('score_musique_entraineur')):
        score += weights['musique_entraineur'] * (row['score_musique_entraineur'] / 10)
    
    if pd.notna(row.get('Gains')):
        score += weights['gains'] * (row['Gains'] / 100000)
    
    if pd.notna(row.get('Ecart_driver')):
        score += weights['ecart_driver'] * (1 / (1 + row['Ecart_driver']))
    
    if pd.notna(row.get('Ecart_entraineur')):
        score += weights['ecart_entraineur'] * (1 / (1 + row['Ecart_entraineur']))
    
    return score

def normaliser_scores(df, score_col='score_brut'):
    min_score = df[score_col].min()
    max_score = df[score_col].max()
    if max_score - min_score > 0:
        df['score_normalise'] = (df[score_col] - min_score) / (max_score - min_score)
    else:
        df['score_normalise'] = 0.5
    return df

def classer_chevaux(df):
    df['score_brut'] = df.apply(calculer_score_cheval, axis=1)
    df = normaliser_scores(df)
    df = df.sort_values('score_normalise', ascending=False).reset_index(drop=True)
    df['rang'] = df.index + 1
    return df

# ------------------------------------------------------------
# Fonctions de pronostics
# ------------------------------------------------------------
def generer_top_3(df):
    top3 = df.head(3)[['N°', 'Cheval', 'score_normalise']].to_dict('records')
    return top3

def generer_bases(df, n=2):
    bases = df.head(n)[['N°', 'Cheval']].to_dict('records')
    return bases

def generer_outsiders(df, n=5, seuil_score=0.3):
    outsiders = df.iloc[3:][df.iloc[3:]['score_normalise'] > seuil_score].head(n)
    return outsiders[['N°', 'Cheval', 'score_normalise']].to_dict('records')

def generer_combinaisons_trio(df, n_combinaisons=10):
    chevaux = df[['N°', 'Cheval', 'score_normalise']].to_dict('records')
    scores = [c['score_normalise'] for c in chevaux]
    total_score = sum(scores)
    probabilities = [s/total_score for s in scores] if total_score > 0 else [1/len(chevaux)]*len(chevaux)
    
    combinaisons = set()
    while len(combinaisons) < n_combinaisons:
        indices = np.random.choice(len(chevaux), size=3, replace=False, p=probabilities)
        comb = tuple(sorted([chevaux[i]['N°'] for i in indices]))
        combinaisons.add(comb)
    
    result = []
    for comb in combinaisons:
        noms = [df[df['N°'] == n]['Cheval'].values[0] for n in comb]
        result.append({
            'combinaison': comb,
            'chevaux': noms
        })
    return result

def generer_combinaisons_quinte(df, n_combinaisons=10):
    chevaux = df[['N°', 'Cheval', 'score_normalise']].to_dict('records')
    scores = [c['score_normalise'] for c in chevaux]
    total_score = sum(scores)
    probabilities = [s/total_score for s in scores] if total_score > 0 else [1/len(chevaux)]*len(chevaux)
    
    combinaisons = set()
    while len(combinaisons) < n_combinaisons:
        indices = np.random.choice(len(chevaux), size=5, replace=False, p=probabilities)
        comb = tuple(sorted([chevaux[i]['N°'] for i in indices]))
        combinaisons.add(comb)
    
    result = []
    for comb in combinaisons:
        noms = [df[df['N°'] == n]['Cheval'].values[0] for n in comb]
        result.append({
            'combinaison': comb,
            'chevaux': noms
        })
    return result

# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------
st.title("🐎 Application de Pronostics Hippiques")
st.markdown("Téléchargez les captures d'écran des statistiques (partants, drivers, entraîneurs, records) pour obtenir une analyse complète.")

uploaded_files = st.file_uploader(
    "📤 Télécharger les photos (PNG, JPG, JPEG)",
    type=['png', 'jpg', 'jpeg'],
    accept_multiple_files=True
)

# Initialisation de l'état de session
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df_final = None

# Aperçu des images
if uploaded_files:
    st.subheader("Aperçu des images téléchargées")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, file in enumerate(uploaded_files):
        with cols[i % 4]:
            image = Image.open(file)
            st.image(image, caption=file.name, use_container_width=True)

# Bouton d'analyse
if st.button("🔍 Analyser la course", type="primary") and uploaded_files:
    with st.spinner("Analyse en cours... Veuillez patienter."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Étape 1: OCR pour chaque image
        status_text.text("Extraction des tableaux par OCR...")
        tables = {'partants': None, 'drivers': None, 'entraineurs': None, 'records': None}
        
        headers_keywords = {
            'partants': ["N°", "Cheval", "Driver", "Entraîneur"],
            'drivers': ["N°", "Cheval", "Driver", "Courses", "Victoires", "Réussite"],
            'entraineurs': ["N°", "Cheval", "Entraîneur", "Courses", "Victoires", "Réussite"],
            'records': ["N°", "Cheval", "Record", "Date"]
        }
        
        for idx, file in enumerate(uploaded_files):
            image = Image.open(file)
            # Essayer de détecter le type de tableau
            for table_type, keywords in headers_keywords.items():
                df = extract_table_from_image(image, keywords)
                if not df.empty and len(df) > 1 and any(k in df.columns for k in keywords):
                    tables[table_type] = df
                    break
            progress_bar.progress((idx+1)/len(uploaded_files))
        
        status_text.text("Nettoyage et structuration des données...")
        time.sleep(1)
        
        # Nettoyage
        for table_type in tables:
            if tables[table_type] is not None:
                tables[table_type] = clean_dataframe(tables[table_type], table_type)
        
        # Fusion
        df_final = fusionner_donnees(
            tables['partants'],
            tables['drivers'],
            tables['entraineurs'],
            tables['records']
        )
        
        if df_final.empty:
            st.error("Aucune donnée valide n'a pu être extraite. Vérifiez vos images.")
        else:
            # Vérifier la présence de la colonne N° et la recréer si nécessaire
            if 'N°' not in df_final.columns:
                st.warning("Le numéro des chevaux (colonne N°) n'a pas été détecté. Utilisation de l'index comme numéro provisoire.")
                df_final['N°'] = range(1, len(df_final) + 1)
            
            # Scoring
            status_text.text("Calcul des scores...")
            df_final = classer_chevaux(df_final)
            
            st.session_state.df_final = df_final
            st.session_state.data_loaded = True
            
            progress_bar.progress(100)
            status_text.text("Analyse terminée!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()

# Affichage des résultats
if st.session_state.data_loaded and st.session_state.df_final is not None:
    df = st.session_state.df_final
    
    st.subheader("📊 Données extraites et scores")
    # Sélectionner les colonnes à afficher
    display_cols = ['N°', 'Cheval', 'Driver', 'Entraîneur', 'Gains', 'Record_secondes',
                    'Reussite_driver', 'Reussite_entraineur', 'score_normalise', 'rang']
    # Garder seulement les colonnes existantes
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols])
    
    # Graphique des scores
    st.subheader("📈 Scores des chevaux")
    fig, ax = plt.subplots(figsize=(10, 6))
    # Utiliser la colonne N° qui est maintenant garantie
    chevaux = df['Cheval'].astype(str) + " (N°" + df['N°'].astype(str) + ")"
    ax.barh(chevaux, df['score_normalise'])
    ax.set_xlabel("Score normalisé")
    ax.set_title("Classement par score")
    st.pyplot(fig)
    
    # Pronostics
    st.subheader("🏆 Pronostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Top 3 probable")
        top3 = generer_top_3(df)
        for i, cheval in enumerate(top3):
            st.write(f"{i+1}. N°{cheval['N°']} - {cheval['Cheval']} (score: {cheval['score_normalise']:.3f})")
    
    with col2:
        st.markdown("#### Bases solides")
        bases = generer_bases(df, 2)
        for cheval in bases:
            st.write(f"🔹 N°{cheval['N°']} - {cheval['Cheval']}")
    
    st.markdown("#### Outsiders intéressants")
    outsiders = generer_outsiders(df, 5)
    for cheval in outsiders:
        st.write(f"👀 N°{cheval['N°']} - {cheval['Cheval']} (score: {cheval['score_normalise']:.3f})")
    
    # Combinaisons Trio
    st.markdown("#### 🎲 10 combinaisons pour le Trio")
    trio_comb = generer_combinaisons_trio(df, 10)
    for i, comb in enumerate(trio_comb):
        st.write(f"{i+1}. {', '.join(comb['chevaux'])} (N°{', '.join(map(str, comb['combinaison']))})")
    
    # Combinaisons Quinté
    st.markdown("#### 🎲 10 combinaisons pour le Quinté+")
    quinte_comb = generer_combinaisons_quinte(df, 10)
    for i, comb in enumerate(quinte_comb):
        st.write(f"{i+1}. {', '.join(comb['chevaux'])} (N°{', '.join(map(str, comb['combinaison']))})")
    
    # Téléchargement CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les données en CSV",
        data=csv,
        file_name='pronostics_hippiques.csv',
        mime='text/csv'
    )

else:
    if uploaded_files:
        st.info("Cliquez sur 'Analyser la course' pour lancer l'extraction.")
    else:
        st.info("Veuillez télécharger des images pour commencer.")
