

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class AdvancedHorseRacingML:
    """
    Modèle ML avancé avec features engineering sophistiquées
    et ensemble de modèles optimisés
    """
    
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        self.feature_names = []
        self.is_trained = False
        
    def create_advanced_features(self, df, race_type):
        """
        Création de features avancées pour le ML
        """
        features_df = df.copy()
        
        # Features de base nettoyées
        base_features = self._get_base_features(features_df)
        
        # Features de forme et historique
        form_features = self._extract_form_features(features_df)
        
        # Features de performance relative
        relative_features = self._calculate_relative_features(features_df)
        
        # Features d'interaction complexes
        interaction_features = self._create_interaction_features(features_df, race_type)
        
        # Features statistiques
        statistical_features = self._compute_statistical_features(features_df)
        
        # Combinaison de toutes les features
        all_features = pd.concat([
            base_features,
            form_features,
            relative_features,
            interaction_features,
            statistical_features
        ], axis=1)
        
        # Nettoyage final
        all_features = all_features.fillna(0)
        self.feature_names = all_features.columns.tolist()
        
        return all_features
    
    def _get_base_features(self, df):
        """Features de base normalisées"""
        base_df = pd.DataFrame()
        
        # Features numériques de base
        if 'odds_numeric' in df.columns:
            base_df['log_odds'] = np.log1p(df['odds_numeric'])
            base_df['inv_odds'] = 1 / (df['odds_numeric'] + 0.1)
            base_df['odds_squared'] = df['odds_numeric'] ** 2
        
        if 'draw_numeric' in df.columns:
            base_df['draw'] = df['draw_numeric']
            base_df['draw_squared'] = df['draw_numeric'] ** 2
            base_df['draw_log'] = np.log1p(df['draw_numeric'])
        
        if 'weight_kg' in df.columns:
            base_df['weight'] = df['weight_kg']
            base_df['weight_normalized'] = (df['weight_kg'] - df['weight_kg'].mean()) / df['weight_kg'].std()
        
        return base_df
    
    def _extract_form_features(self, df):
        """Analyse poussée de la forme récente"""
        form_df = pd.DataFrame()
        
        if 'Musique' in df.columns:
            for i, row in df.iterrows():
                musique = str(row['Musique']) if pd.notna(row['Musique']) else ""
                
                # Extraction des positions
                positions = [int(c) for c in musique if c.isdigit()]
                
                if positions:
                    # Statistiques de forme
                    form_df.loc[i, 'avg_position'] = np.mean(positions)
                    form_df.loc[i, 'best_position'] = min(positions)
                    form_df.loc[i, 'worst_position'] = max(positions)
                    form_df.loc[i, 'position_std'] = np.std(positions) if len(positions) > 1 else 0
                    
                    # Progression/régression
                    if len(positions) >= 3:
                        recent_trend = np.polyfit(range(len(positions[-3:])), positions[-3:], 1)[0]
                        form_df.loc[i, 'recent_trend'] = recent_trend
                    
                    # Constance
                    form_df.loc[i, 'consistency'] = 1 / (1 + np.std(positions))
                    
                    # Victoires et places récentes
                    form_df.loc[i, 'recent_wins'] = sum(1 for p in positions[:3] if p == 1)
                    form_df.loc[i, 'recent_places'] = sum(1 for p in positions[:3] if p <= 3)
                    
                    # Séries
                    current_streak = 0
                    for p in positions:
                        if p <= 3:
                            current_streak += 1
                        else:
                            break
                    form_df.loc[i, 'place_streak'] = current_streak
        
        return form_df.fillna(0)
    
    def _calculate_relative_features(self, df):
        """Features relatives à la course"""
        rel_df = pd.DataFrame()
        
        if 'odds_numeric' in df.columns:
            # Position relative dans les cotes
            rel_df['odds_rank'] = df['odds_numeric'].rank()
            rel_df['odds_percentile'] = df['odds_numeric'].rank(pct=True)
            
            # Écart à la moyenne
            odds_mean = df['odds_numeric'].mean()
            rel_df['odds_vs_avg'] = df['odds_numeric'] - odds_mean
            rel_df['odds_ratio_avg'] = df['odds_numeric'] / odds_mean
        
        if 'weight_kg' in df.columns:
            # Position relative dans les poids
            rel_df['weight_rank'] = df['weight_kg'].rank()
            rel_df['weight_percentile'] = df['weight_kg'].rank(pct=True)
            
            weight_mean = df['weight_kg'].mean()
            rel_df['weight_vs_avg'] = df['weight_kg'] - weight_mean
        
        if 'draw_numeric' in df.columns:
            # Position relative des numéros
            rel_df['draw_percentile'] = df['draw_numeric'].rank(pct=True)
        
        return rel_df.fillna(0)
    
    def _create_interaction_features(self, df, race_type):
        """Features d'interaction spécialisées"""
        inter_df = pd.DataFrame()
        
        # Interactions de base
        if 'odds_numeric' in df.columns and 'draw_numeric' in df.columns:
            inter_df['odds_draw_product'] = df['odds_numeric'] * df['draw_numeric']
            inter_df['odds_draw_ratio'] = df['odds_numeric'] / (df['draw_numeric'] + 1)
        
        if 'odds_numeric' in df.columns and 'weight_kg' in df.columns:
            inter_df['odds_weight_product'] = df['odds_numeric'] * df['weight_kg']
            inter_df['weight_per_odds'] = df['weight_kg'] / (df['odds_numeric'] + 1)
        
        # Interactions spécifiques au type de course
        if race_type == "PLAT":
            # Avantage corde pour les petits numéros
            if 'draw_numeric' in df.columns:
                inter_df['inner_draw_bonus'] = np.where(df['draw_numeric'] <= 4, 1, 0)
                inter_df['outer_draw_penalty'] = np.where(df['draw_numeric'] >= 12, 1, 0)
            
            # Pénalité poids lourd
            if 'weight_kg' in df.columns:
                weight_median = df['weight_kg'].median()
                inter_df['heavy_weight_penalty'] = np.where(
                    df['weight_kg'] > weight_median + 2, 1, 0
                )
        
        elif race_type in ["ATTELE_AUTOSTART"]:
            # Bonus numéros optimaux attelé
            if 'draw_numeric' in df.columns:
                inter_df['optimal_draw_attele'] = np.where(
                    df['draw_numeric'].isin([4, 5, 6]), 1, 0
                )
                inter_df['bad_draw_attele'] = np.where(
                    df['draw_numeric'].isin([1, 2, 3]) | (df['draw_numeric'] >= 10), 1, 0
                )
        
        return inter_df.fillna(0)
    
    def _compute_statistical_features(self, df):
        """Features statistiques avancées"""
        stat_df = pd.DataFrame()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in ['odds_numeric', 'draw_numeric', 'weight_kg']:
            if col in numeric_cols:
                values = df[col]
                
                # Z-score
                stat_df[f'{col}_zscore'] = (values - values.mean()) / values.std()
                
                # Percentiles
                stat_df[f'{col}_is_top25'] = (values <= values.quantile(0.25)).astype(int)
                stat_df[f'{col}_is_bottom25'] = (values >= values.quantile(0.75)).astype(int)
                
                # Outliers
                Q1, Q3 = values.quantile(0.25), values.quantile(0.75)
                IQR = Q3 - Q1
                stat_df[f'{col}_is_outlier'] = (
                    (values < Q1 - 1.5 * IQR) | (values > Q3 + 1.5 * IQR)
                ).astype(int)
        
        return stat_df.fillna(0)
    
    def build_ensemble_models(self):
        """Construction d'un ensemble de modèles optimisés"""
        
        # Modèles de base avec hyperparamètres optimisés
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                random_state=42
            ),
            
            'ridge': Ridge(alpha=1.0),
            
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            ),
            
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        
        # Modèle d'ensemble
        self.ensemble_model = VotingRegressor([
            ('rf', self.models['random_forest']),
            ('gb', self.models['gradient_boosting']),
            ('ridge', self.models['ridge'])
        ])
    
    def optimize_hyperparameters(self, X, y, model_name='random_forest'):
        """Optimisation des hyperparamètres par GridSearch"""
        
        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [8, 10, 12],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8]
            }
            base_model = GradientBoostingRegressor(random_state=42)
        
        else:
            return self.models[model_name]  # Retour du modèle par défaut
        
        # GridSearch avec validation croisée temporelle
        tscv = TimeSeriesSplit(n_splits=3)
        grid_search = GridSearchCV(
            base_model, param_grid, 
            cv=tscv, scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def train_advanced_models(self, X, y, optimize_hyperparams=False):
        """Entraînement des modèles avancés"""
        
        if len(X) < 10:
            raise ValueError("Pas assez de données pour l'entraînement avancé")
        
        # Preprocessing avancé
        X_scaled = self.scaler.fit_transform(X)
        
        # Features polynomiales (degré 2)
        if X.shape[1] <= 20:  # Seulement si pas trop de features
            X_poly = self.poly_features.fit_transform(X_scaled)
        else:
            X_poly = X_scaled
        
        # Construction des modèles
        self.build_ensemble_models()
        
        # Optimisation optionnelle
        if optimize_hyperparams:
            self.models['random_forest'] = self.optimize_hyperparameters(
                X_poly, y, 'random_forest'
            )
            self.models['gradient_boosting'] = self.optimize_hyperparameters(
                X_poly, y, 'gradient_boosting'
            )
        
        # Entraînement de tous les modèles
        results = {}
        for name, model in self.models.items():
            try:
                model.fit(X_poly, y)
                y_pred = model.predict(X_poly)
                
                results[name] = {
                    'mse': mean_squared_error(y, y_pred),
                    'mae': mean_absolute_error(y, y_pred),
                    'r2': model.score(X_poly, y)
                }
                
                # Feature importance si disponible
                if hasattr(model, 'feature_importances_'):
                    feature_names = (self.poly_features.get_feature_names_out(self.feature_names) 
                                   if hasattr(self.poly_features, 'get_feature_names_out') 
                                   else self.feature_names)
                    results[name]['feature_importance'] = dict(
                        zip(feature_names[:len(model.feature_importances_)], 
                            model.feature_importances_)
                    )
                    
            except Exception as e:
                print(f"Erreur entraînement {name}: {e}")
                continue
        
        # Entraînement du modèle d'ensemble
        try:
            self.ensemble_model.fit(X_poly, y)
            y_pred_ensemble = self.ensemble_model.predict(X_poly)
            
            results['ensemble'] = {
                'mse': mean_squared_error(y, y_pred_ensemble),
                'mae': mean_absolute_error(y, y_pred_ensemble),
                'r2': self.ensemble_model.score(X_poly, y)
            }
            
        except Exception as e:
            print(f"Erreur ensemble: {e}")
        
        self.is_trained = True
        return results
    
    def predict_advanced(self, X, model_name='ensemble'):
        """Prédictions avec le modèle sélectionné"""
        if not self.is_trained:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.poly_features, 'transform'):
            X_poly = self.poly_features.transform(X_scaled)
        else:
            X_poly = X_scaled
        
        if model_name == 'ensemble' and self.ensemble_model:
            return self.ensemble_model.predict(X_poly)
        elif model_name in self.models:
            return self.models[model_name].predict(X_poly)
        else:
            # Fallback sur Random Forest
            return self.models['random_forest'].predict(X_poly)
    
    def get_prediction_confidence(self, X, n_bootstrap=50):
        """Calcul de l'intervalle de confiance des prédictions"""
        if not self.is_trained:
            return np.zeros(len(X)), np.zeros(len(X))
        
        predictions = []
        
        # Bootstrap predictions avec différents modèles
        for _ in range(n_bootstrap):
            # Sélection aléatoire d'un modèle
            model_name = np.random.choice(list(self.models.keys()))
            pred = self.predict_advanced(X, model_name)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calcul des intervalles de confiance
        lower_bound = np.percentile(predictions, 25, axis=0)
        upper_bound = np.percentile(predictions, 75, axis=0)
        
        return lower_bound, upper_bound
Response
Created file /home/user/advanced_ml_models.py (15965 characters)
