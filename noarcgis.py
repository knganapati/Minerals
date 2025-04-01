# Run the complete prediction pipeline
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import seaborn as sns
from sentence_transformers import SentenceTransformer
import faiss
import geopandas as gpd
from shapely.geometry import Point
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MineralPredictor:
    def __init__(self):
        """Initialize the advanced mineral prediction model"""
        # Data components
        self.kb_data = None
        self.minerals = []
        self.feature_cols = []
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        # Models
        self.xgb_model = None
        self.neural_model = None
        self.lgbm_model = None
        self.ensemble_weights = [0.5, 0.3, 0.2]  # XGBoost, Neural Network, LightGBM
        
        # RAG system
        self.embedder = SentenceTransformer('all-mpnet-base-v2')  # Upgraded model
        self.kb_index = None
        self.kb_ids = None
        
        # Flag for trained state
        self.is_trained = False
    
    def load_data(self, excel_path):
        """Load and process the mineral knowledge base"""
        print(f"Loading knowledge base from {excel_path}")
        import pandas as pd
        import re
        import os
#  Read Excel without headers
        df = pd.read_excel(excel_path, header=None)
        print(df.head(4))
        # Find mineral sections
        # section_rows=[0, 112, 156, 243, 281, 453, 709, 809, 813]
        # section_rows = []
        # for i, row in df.iterrows():
        #     if isinstance(row[0], str) and '.' in row[0] and '(' in row[0] and ')' in row[0]:
        #         section_rows.append(i)
        section_rows = []
        for i, row in df.iterrows():
            # Check if the first column is a string and contains the characters '.' and '(' and ')'
            if isinstance(row[0], str) and '.' in row[0] and '(' in row[0] and ')' in row[0]:
                section_rows.append(i)
            # Check if the first column contains more than 20 words
            elif isinstance(row[0], str) and len(row[0].split()) > 20:
                section_rows.append(i)

        # Process each section
        all_data = []

 
        # output_dir = "./output_minerals"
        # os.makedirs(output_dir, exist_ok=True)

     
        # # Find mineral sections
        # section_rows = []
        # for i, row in df.iterrows():
        #     if isinstance(row[0], str) and '.' in row[0] and '(' in row[0] and ')' in row[0]:
        #         section_rows.append(i)
        #     elif isinstance(row[0], str) and len(row[0].split()) > 20:
        #         section_rows.append(i)

        # for i in range(len(section_rows)):
        #     start = section_rows[i]
        #     end = section_rows[i + 1] if i + 1 < len(section_rows) else len(df)

        #     # Get the mineral description text
        #     mineral_header = df.iloc[start, 0]

        #     # Extract mineral name
        #     match = re.match(r'^[A-Z][a-z]+(_[A-Z][a-z]+)*', mineral_header)
        #     mineral_name = match.group(0) if match else mineral_header.split('(')[0].strip()
        #     mineral_name = mineral_name.replace(" ", "_")  # Ensure valid filename

        #     # Find tables in section
        #     section_data = df.iloc[start:end]
        #     table_rows = [j for j, row in section_data.iterrows() if row[0] == "Deposit Name"]
           

        #     all_tables = []
        #     for table_row in table_rows:
        #         header_row = df.iloc[table_row]
        #         headers = [h for h in header_row if pd.notnull(h)]

        #         table_end = next((j for j in range(table_row + 1, end) 
        #                         if pd.isnull(df.iloc[j, 0]) or (
        #                             j + 1 < len(df) and isinstance(df.iloc[j+1, 0], str) and 
        #                             '.' in str(df.iloc[j+1, 0]) and '(' in str(df.iloc[j+1, 0])
        #                         )), end)

        #         table_data = df.iloc[table_row+1:table_end]
                
        #         if len(headers) > table_data.shape[1]:
        #             headers = headers[:table_data.shape[1]]

        #         table_df = pd.DataFrame(table_data.values, columns=headers)
        #         table_df['Mineral'] = mineral_name
        #         all_tables.append(table_df)

        #     if all_tables:
        #         mineral_df = pd.concat(all_tables, ignore_index=True)
        #         mineral_df.to_csv(f"{output_dir}/{mineral_name}.csv", index=False)
##

        import re

        print("section_rows", section_rows)

        for i in range(len(section_rows)):
            start = section_rows[i]
            end = section_rows[i + 1] if i + 1 < len(section_rows) else len(df)
            
            # Get the mineral description text
            mineral_header = df.iloc[start, 0]

            # Use regex to extract the mineral name (first word with a capital letter)
            match = re.match(r'^[A-Z][a-z]+(_[A-Z][a-z]+)*', mineral_header)

            
            if match:
                mineral_name = match.group(0)
            else:
                mineral_name = mineral_header.split('(')[0].strip()  # fallback if regex fails
            
            # Continue processing mineral_name
            print(f"Mineral Name: {mineral_name}")

            # Find tables in section
            section_data = df.iloc[start:end]
            table_rows = []
            
            for j, row in section_data.iterrows():
                if row[0] == "Deposit Name":
                    table_rows.append(j)
            
            # Process each table
            for table_row in table_rows:
                # Extract headers
                header_row = df.iloc[table_row]
                headers = [h for h in header_row if pd.notnull(h)]
                
                # Find table end
                table_end = end
                for j in range(table_row + 1, end):
                    if pd.isnull(df.iloc[j, 0]) or (j + 1 < len(df) and isinstance(df.iloc[j+1, 0], str) and 
                                                    ('.' in str(df.iloc[j+1, 0]) and '(' in str(df.iloc[j+1, 0]))):
                        table_end = j
                        break
                
                # Extract table data
                table_data = df.iloc[table_row+1:table_end]
                
                # Ensure we have the right number of columns
                if len(headers) > table_data.shape[1]:
                    headers = headers[:table_data.shape[1]]
                
                # Convert to DataFrame with proper headers
                table_df = pd.DataFrame(table_data.values, columns=headers + [f"Extra_{i}" for i in range(table_data.shape[1] - len(headers))])
                table_df = table_df[headers].copy()
                
                # Add mineral name
                table_df['Mineral'] = mineral_name
                
                # Add to collection
                all_data.append(table_df)
        print("all_data", all_data)
        # Combine all data
        if all_data:
            self.kb_data = pd.concat(all_data, ignore_index=True)
            self.minerals = self.kb_data['Mineral'].unique().tolist()
            print(f"Loaded {len(self.kb_data)} deposits for {len(self.minerals)} minerals")
        
        
        
        # print("Extracted table data:")
        # print(table_data.head())  #edited 
        # if not table_data.empty:

        #    self.kb_data = pd.concat(all_tables, ignore_index=True) if all_tables else pd.DataFrame()
        #    print(f"Columns in extracted table: {table_df.columns}")
        #    self.minerals = self.kb_data['Mineral'].unique().tolist()
        #    print(f"Loaded {len(self.kb_data)} deposits for {len(self.minerals)} minerals")
            
            
            
        else:
            print("No data tables found in the file")
            self.kb_data = pd.DataFrame()
        print("self.kb_data", self.kb_data)
        return self.kb_data
    
    def build_advanced_rag(self, chunk_size=3):
        """Build an advanced RAG system with overlapping chunks for better context"""
        if self.kb_data is None or len(self.kb_data) == 0:
            raise ValueError("Knowledge base not loaded. Call load_data first.")
        
        print("Building advanced RAG system...")
        
        # Create better text representation with more structure
        all_chunks = []
        self.kb_ids = []
        
        # For each mineral type
        for mineral in self.minerals:
            mineral_data = self.kb_data[self.kb_data['Mineral'] == mineral].copy()
            
            # Create a mineral overview chunk
            mineral_overview = f"MINERAL: {mineral}\n\n"
            mineral_overview += f"DEPOSIT COUNT: {len(mineral_data)}\n"
            
            # Add common deposit types for this mineral
            if 'Type of Deposit' in mineral_data.columns:
                deposit_types = mineral_data['Type of Deposit'].value_counts().to_dict()
                mineral_overview += "COMMON DEPOSIT TYPES:\n"
                for dtype, count in deposit_types.items():
                    if pd.notnull(dtype) and dtype:
                        mineral_overview += f"- {dtype}: {count} deposits\n"
            
            # Add common host rocks for this mineral
            if 'Host Rocks' in mineral_data.columns:
                host_rocks = []
                for rocks in mineral_data['Host Rocks']:
                    if pd.notnull(rocks) and rocks:
                        host_rocks.extend([r.strip() for r in rocks.split(',')])
                
                if host_rocks:
                    rock_counts = pd.Series(host_rocks).value_counts().to_dict()
                    mineral_overview += "COMMON HOST ROCKS:\n"
                    for rock, count in list(rock_counts.items())[:5]:
                        mineral_overview += f"- {rock}: {count} occurrences\n"
            
            # Add this overview chunk
            all_chunks.append(mineral_overview)
            self.kb_ids.append({"type": "mineral_overview", "mineral": mineral, "id": -1})
            
            # Create chunks for deposit data
            deposits = mineral_data['Deposit Name'].tolist()
            
            # Process deposits in overlapping chunks for better context
            for i in range(0, len(deposits), chunk_size):
                chunk_deposits = deposits[i:i+chunk_size]
                chunk_text = f"MINERAL: {mineral}\n\n"
                
                # Add each deposit's details
                for deposit in chunk_deposits:
                    deposit_data = mineral_data[mineral_data['Deposit Name'] == deposit].iloc[0]
                    chunk_text += f"DEPOSIT: {deposit}\n"
                    
                    # Add all available fields
                    for col in deposit_data.index:
                        if col not in ['Mineral', 'Deposit Name'] and pd.notnull(deposit_data[col]):
                            chunk_text += f"{col}: {deposit_data[col]}\n"
                    
                    chunk_text += "\n"
                
                # Add this deposit chunk
                all_chunks.append(chunk_text)
                self.kb_ids.append({"type": "deposit_chunk", "mineral": mineral, "deposits": chunk_deposits})
        
        # Generate embeddings
        print(f"Creating embeddings for {len(all_chunks)} document chunks...")
        embeddings = self.embedder.encode(all_chunks, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.kb_index = faiss.IndexFlatL2(dimension)
        self.kb_index.add(np.array(embeddings).astype('float32'))
        
        print(f"Advanced RAG system built with {len(all_chunks)} document chunks")
        return self.kb_index
    
    def query_rag(self, query_text, k=7):
        """Query the RAG system with advanced processing"""
        if self.kb_index is None:
            raise ValueError("RAG system not built. Call build_advanced_rag first.")
        
        # Encode query
        query_embedding = self.embedder.encode([query_text])
        
        # Search the index
        distances, indices = self.kb_index.search(np.array(query_embedding).astype('float32'), k)
        
        # Process results
        results = []
        mineral_scores = {}
        deposit_scores = {}
        feature_scores = {}
        
        for i, idx in enumerate(indices[0]):
            # Get relevance score (inverse of distance)
            relevance = 1.0 / (1.0 + distances[0][i])
            
            # Get chunk metadata
            chunk_meta = self.kb_ids[idx]
            
            # Add to mineral scores
            mineral = chunk_meta.get('mineral', '')
            if mineral:
                mineral_scores[mineral] = mineral_scores.get(mineral, 0) + relevance
            
            # If it's a deposit chunk, extract deposit data
            if chunk_meta['type'] == 'deposit_chunk':
                # Get deposits
                deposits = chunk_meta.get('deposits', [])
                
                # Add to deposit scores
                for deposit in deposits:
                    deposit_scores[deposit] = deposit_scores.get(deposit, 0) + relevance
                
                # Get deposit data for feature extraction
                for deposit in deposits:
                    deposit_data = self.kb_data[(self.kb_data['Mineral'] == mineral) & 
                                                (self.kb_data['Deposit Name'] == deposit)]
                    
                    if not deposit_data.empty:
                        deposit_row = deposit_data.iloc[0]
                        
                        # Extract features like deposit type, host rocks, etc.
                        for col in ['Type of Deposit', 'Host Rocks', 'Mineralogy', 'Tectonic Settings']:
                            if col in deposit_row and pd.notnull(deposit_row[col]):
                                val = deposit_row[col]
                                if isinstance(val, str):
                                    for item in [x.strip() for x in val.split(',')]:
                                        feature_key = f"{col}_{item}"
                                        feature_scores[feature_key] = feature_scores.get(feature_key, 0) + relevance
            
            results.append((idx, distances[0][i], chunk_meta))
        
        # Return processed results
        return {
            'raw_results': results,
            'mineral_scores': mineral_scores,
            'deposit_scores': deposit_scores,
            'feature_scores': feature_scores
        }
    
    def extract_advanced_features(self, lat, lon):
        """Extract richer features for a location using the advanced RAG system"""
        # Create a detailed query with geological context
        query = f"""
        Geological analysis for coordinates: latitude {lat}, longitude {lon}.
        What minerals, deposit types, host rocks, and geological features are likely to be found here?
        What is the mineralogy, tectonic setting, and ore genesis that might be present?
        """
        
        # Get RAG results
        rag_results = self.query_rag(query, k=10)
        
        # Initialize feature dictionary
        features = {
            'lat': lat,
            'lon': lon,
            'lat_sin': np.sin(np.radians(lat)),
            'lat_cos': np.cos(np.radians(lat)),
            'lon_sin': np.sin(np.radians(lon)),
            'lon_cos': np.cos(np.radians(lon))
        }
        
        # Add mineral scores
        for mineral, score in rag_results['mineral_scores'].items():
            features[f'mineral_{mineral}'] = score
        
        # Add geological feature scores
        for feature, score in rag_results['feature_scores'].items():
            # Clean feature name for use as column name
            feature_clean = feature.replace(' ', '_').replace('-', '_').replace('/', '_').lower()
            features[feature_clean] = score
        
        # Ensure all minerals have a feature, even if zero
        for mineral in self.minerals:
            key = f'mineral_{mineral}'
            if key not in features:
                features[key] = 0.0
        
        return features
    
    def create_training_data(self, synthetic_multiplier=3, negative_ratio=0.5):
        """Create enhanced training data with more synthetic examples and better balance"""
        if self.kb_data is None or len(self.kb_data) == 0:
            raise ValueError("Knowledge base not loaded. Call load_data first.")
        
        if self.kb_index is None:
            raise ValueError("RAG system not built. Call build_advanced_rag first.")
        
        print("Creating enhanced training dataset...")
        
        # Create synthetic training examples
        training_data = []
        
        # Group by mineral type
        for mineral, group in self.kb_data.groupby('Mineral'):
            print(f"Generating examples for {mineral}...")
            
            # For each deposit of this mineral type
            for _, deposit in tqdm(group.iterrows(), total=len(group)):
                location = deposit.get('Location', '')
                
                # Create a base location 
                # In a real system, you would geocode the actual location
                # Here we're generating random coordinates for demonstration
                base_lat = np.random.uniform(-60, 60)
                base_lon = np.random.uniform(-180, 180)
                
                # Create positive examples (with this mineral)
                for _ in range(synthetic_multiplier):
                    # Add some noise to coordinates to create realistic clusters
                    lat = base_lat + np.random.normal(0, 0.5)
                    lon = base_lon + np.random.normal(0, 0.5)
                    
                    # Extract features
                    features = self.extract_advanced_features(lat, lon)
                    
                    # Add target
                    features['target_mineral'] = mineral
                    
                    training_data.append(features)
                
                # Create negative examples (with other minerals)
                other_minerals = [m for m in self.minerals if m != mineral]
                if other_minerals and negative_ratio > 0:
                    for _ in range(int(synthetic_multiplier * negative_ratio)):
                        # More distant coordinates
                        lat = base_lat + np.random.normal(0, 5.0)
                        lon = base_lon + np.random.normal(0, 5.0)
                        
                        # Extract features
                        features = self.extract_advanced_features(lat, lon)
                        
                        # Assign random other mineral
                        features['target_mineral'] = np.random.choice(other_minerals)
                        
                        training_data.append(features)
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data)
        
        print(f"Created {len(df)} training examples")
        
        # Ensure all examples have the same columns by filling missing values
        all_columns = set()
        for example in training_data:
            all_columns.update(example.keys())
        
        # Remove target from feature columns
        self.feature_cols = [col for col in all_columns if col != 'target_mineral']
        
        # Fill missing values
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        return df
    
    def build_neural_network(self, input_dim, num_classes):
        """Build a neural network for mineral classification"""
        # Input layer
        inputs = Input(shape=(input_dim,))
        
        # Hidden layers with batch normalization and dropout for regularization
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.1)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_advanced_models(self, cross_val=True, n_folds=5):
        """Train the ensemble of models with cross-validation"""
        print("Training advanced mineral prediction models...")
        
        # Create training data
        training_df = self.create_training_data()
        
        # Prepare features and target
        X = training_df[self.feature_cols]
        y = training_df['target_mineral']
        
        # Convert mineral names to indices
        mineral_to_idx = {mineral: i for i, mineral in enumerate(self.minerals)}
        y_idx = y.map(mineral_to_idx)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        if cross_val:
            # Use cross-validation for better evaluation
            print(f"Performing {n_folds}-fold cross-validation...")
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            nn_val_scores = []
            xgb_val_scores = []
            lgbm_val_scores = []
            ensemble_val_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_idx)):
                print(f"Training fold {fold+1}/{n_folds}...")
                
                # Split data
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_idx.iloc[train_idx], y_idx.iloc[val_idx]
                
                # Train neural network
                nn_model = self.build_neural_network(X_train.shape[1], len(self.minerals))
                nn_model.fit(
                    X_train, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    verbose=0,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            restore_best_weights=True
                        )
                    ]
                )
                
                # Train XGBoost
                xgb_model = XGBClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                xgb_model.fit(X_train, y_train)
                
                # Train LightGBM
                lgbm_model = LGBMClassifier(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                lgbm_model.fit(X_train, y_train)
                
                # Make predictions
                nn_pred = np.argmax(nn_model.predict(X_val), axis=1)
                xgb_pred = xgb_model.predict(X_val)
                lgbm_pred = lgbm_model.predict(X_val)
                
                # Evaluate individual models
                nn_score = accuracy_score(y_val, nn_pred)
                xgb_score = accuracy_score(y_val, xgb_pred)
                lgbm_score = accuracy_score(y_val, lgbm_pred)
                
                nn_val_scores.append(nn_score)
                xgb_val_scores.append(xgb_score)
                lgbm_val_scores.append(lgbm_score)
                
                # Ensemble prediction (weighted voting)
                nn_proba = nn_model.predict(X_val)
                xgb_proba = xgb_model.predict_proba(X_val)
                lgbm_proba = lgbm_model.predict_proba(X_val)
                
                ensemble_proba = (
                    self.ensemble_weights[0] * xgb_proba +
                    self.ensemble_weights[1] * nn_proba +
                    self.ensemble_weights[2] * lgbm_proba
                )
                ensemble_pred = np.argmax(ensemble_proba, axis=1)
                ensemble_score = accuracy_score(y_val, ensemble_pred)
                ensemble_val_scores.append(ensemble_score)
                
                print(f"Fold {fold+1} - Neural Network: {nn_score:.4f}, XGBoost: {xgb_score:.4f}, LightGBM: {lgbm_score:.4f}, Ensemble: {ensemble_score:.4f}")
            
            # Print validation results
            print("\nCross-validation results:")
            print(f"Neural Network: {np.mean(nn_val_scores):.4f} ± {np.std(nn_val_scores):.4f}")
            print(f"XGBoost: {np.mean(xgb_val_scores):.4f} ± {np.std(xgb_val_scores):.4f}")
            print(f"LightGBM: {np.mean(lgbm_val_scores):.4f} ± {np.std(lgbm_val_scores):.4f}")
            print(f"Ensemble: {np.mean(ensemble_val_scores):.4f} ± {np.std(ensemble_val_scores):.4f}")
        
        # Train final models on all data
        print("Training final models on all data...")
        
        # Neural Network
        self.neural_model = self.build_neural_network(X_scaled.shape[1], len(self.minerals))
        self.neural_model.fit(
            X_scaled, y_idx,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # XGBoost
        self.xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.xgb_model.fit(X_scaled, y_idx)
        
        # LightGBM
        self.lgbm_model = LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.lgbm_model.fit(X_scaled, y_idx)
        
        # Get feature importance from XGBoost
        importance = self.xgb_model.feature_importances_
        self.feature_importance = sorted(zip(self.feature_cols, importance), key=lambda x: x[1], reverse=True)
        
        print("Top 10 most important features:")
        for feature, importance in self.feature_importance[:10]:
            print(f"- {feature}: {importance:.4f}")
        
        self.is_trained = True
        return {
            'neural_network': self.neural_model,
            'xgboost': self.xgb_model,
            'lightgbm': self.lgbm_model,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, lat, lon):
        """Make predictions using the ensemble of models"""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train_advanced_models first.")
        
        # Extract features
        features = self.extract_advanced_features(lat, lon)
        
        # Convert to array matching the training features
        feature_arr = np.array([[features.get(col, 0) for col in self.feature_cols]])
        
        # Scale features
        feature_scaled = self.scaler.transform(feature_arr)
        
        # Get predictions from each model
        nn_proba = self.neural_model.predict(feature_scaled)[0]
        xgb_proba = self.xgb_model.predict_proba(feature_scaled)[0]
        lgbm_proba = self.lgbm_model.predict_proba(feature_scaled)[0]
        
        # Ensemble prediction
        ensemble_proba = (
            self.ensemble_weights[0] * xgb_proba +
            self.ensemble_weights[1] * nn_proba +
            self.ensemble_weights[2] * lgbm_proba
        )
        
        # Get top minerals
        top_indices = np.argsort(ensemble_proba)[::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            mineral = self.minerals[idx]
            probability = float(ensemble_proba[idx])
            results.append((mineral, probability))
        
        # Return prediction with model-specific probabilities
        return {
            'top_mineral': results[0][0],
            'top_probability': results[0][1],
            'predictions': results,
            'model_probabilities': {
                'neural_network': {self.minerals[i]: float(nn_proba[i]) for i in range(len(self.minerals))},
                'xgboost': {self.minerals[i]: float(xgb_proba[i]) for i in range(len(self.minerals))},
                'lightgbm': {self.minerals[i]: float(lgbm_proba[i]) for i in range(len(self.minerals))}
            },
            'coordinates': {
                'latitude': lat,
                'longitude': lon
            }
        }
    
    def predict_region(self, min_lat, max_lat, min_lon, max_lon, grid_size=20):
        """Predict minerals across a region with higher resolution"""
        # Create a grid
        lats = np.linspace(min_lat, max_lat, grid_size)
        lons = np.linspace(min_lon, max_lon, grid_size)
        
        # Make predictions
        print(f"Predicting minerals on a {grid_size}x{grid_size} grid...")
        results = []
        
        for lat in tqdm(lats):
            for lon in lons:
                pred = self.predict(lat, lon)
                results.append({
                    'lat': lat,
                    'lon': lon,
                    'prediction': pred
                })
        
        return results
    
    def create_3d_map(self, predictions, output_dir, project_name="MineralPredictions"):
        """Create a high-quality 3D map using matplotlib"""
        # Create output directory if needed
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract prediction data
        points = []
        properties = []
        
        for p in predictions:
            # Create point
            point = Point(p['lon'], p['lat'])
            points.append(point)
            
            # Create properties
            prop = {
                'lat': p['lat'],
                'lon': p['lon'],
                'mineral': p['prediction']['top_mineral'],
                'probability': p['prediction']['top_probability']
            }
            
            # Add top 3 minerals
            for i, (mineral, prob) in enumerate(p['prediction']['predictions'][:3]):
                prop[f'mineral_{i+1}'] = mineral
                prop[f'prob_{i+1}'] = prob
            
            properties.append(prop)
        # print("properties")    
        # print(properties)
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(properties, geometry=points, crs="EPSG:4326")
        print(gdf)
        # Save as GeoJSON
        geojson_path = os.path.join(output_dir, f"{project_name}.geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")
        
        # Get all unique minerals
        all_minerals = set()
        for p in properties:
            all_minerals.add(p['mineral'])
        # print(all_minerals)
        # Create individual files for each mineral
        for mineral in all_minerals:
            # Filter for this mineral
            mineral_gdf = gdf[gdf['mineral'] == mineral].copy()
            
            if len(mineral_gdf) > 0:
                # Save as separate GeoJSON
                mineral_path = os.path.join(output_dir, f"{mineral.replace(' ', '_')}.geojson")
                mineral_gdf.to_file(mineral_path, driver="GeoJSON")
        
        # Create a 3D visualization with contours
        # Create a grid for interpolation
        print("Creating 3D visualization...")
        
        # Get bounding box
        min_lon = min(p['lon'] for p in properties)
        max_lon = max(p['lon'] for p in properties)
        min_lat = min(p['lat'] for p in properties)
        max_lat = max(p['lat'] for p in properties)
        
        # Add some padding
        lon_pad = (max_lon - min_lon) * 0.1
        lat_pad = (max_lat - min_lat) * 0.1
        
        min_lon -= lon_pad
        max_lon += lon_pad
        min_lat -= lat_pad
        max_lat += lat_pad
        
        # Create meshgrid
        grid_size = 100
        lon_grid = np.linspace(min_lon, max_lon, grid_size)
        lat_grid = np.linspace(min_lat, max_lat, grid_size)
        
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Create multiple figures - one per top mineral
        top_minerals = gdf['mineral'].value_counts().index.tolist()
        
        for i, mineral in enumerate(top_minerals):
            # Create 3D figure
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Filter points for this mineral
            mineral_df = gdf[gdf['mineral'] == mineral]
            
            # Extract coordinates and probabilities
            x = mineral_df['lon'].values
            y = mineral_df['lat'].values
            z = mineral_df['probability'].values * 10  # Scale for visibility
            
            # Plot points
            scatter = ax.scatter(
                x, y, z,
                c=z,
                cmap='viridis',
                alpha=0.7,
                s=50,
                edgecolors='k',
                linewidths=0.5
            )
            
            # Set labels and title
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Probability')
            ax.set_title(f'3D Mineral Prediction Map: {mineral}')
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Probability')
            
            # Save figure
            fig_path = os.path.join(output_dir, f"{project_name}_{mineral.replace(' ', '_')}.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create a composite figure with all minerals
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a colormap for each mineral
        mineral_colors = plt.cm.tab10(np.linspace(0, 1, len(all_minerals)))
        mineral_color_dict = {m: mineral_colors[i] for i, m in enumerate(all_minerals)}
        
        # Plot points for each mineral
        for mineral in all_minerals:
            mineral_df = gdf[gdf['mineral'] == mineral]
            
            x = mineral_df['lon'].values
            y = mineral_df['lat'].values
            z = mineral_df['probability'].values * 10
            
            ax.scatter(
                x, y, z,
                label=mineral,
                color=mineral_color_dict[mineral],
                alpha=0.7,
                s=40,
                edgecolors='k',
                linewidths=0.5
            )
        
        # Set labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('Probability')
        ax.set_title('3D Mineral Prediction Map: All Minerals')
        
        # Add legend
        ax.legend()
        
        # Save composite figure
        composite_path = os.path.join(output_dir, f"{project_name}_composite.png")
        plt.savefig(composite_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create heatmap visualizations for top minerals
        print("Creating mineral heatmaps...")
        for mineral in top_minerals:
            # Create 2D figure
            plt.figure(figsize=(12, 10))
            
            # Filter points for this mineral
            mineral_df = gdf[gdf['mineral'] == mineral]
            
            # Create heatmap
            plt.scatter(
                mineral_df['lon'], 
                mineral_df['lat'], 
                c=mineral_df['probability'],
                cmap='hot',
                alpha=0.8,
                s=100,
                edgecolors='k',
                linewidths=0.5
            )
            
            # Add basemap (if contextily is available)
            try:
                import contextily as ctx
                ctx.add_basemap(ax=plt.gca(), crs=gdf.crs.to_string())
            except:
                pass  # Skip if contextily is not available
            
            # Set labels and title
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title(f'Heatmap for {mineral}')
            plt.colorbar(label='Probability')
            
            # Save figure
            heatmap_path = os.path.join(output_dir, f"{project_name}_{mineral.replace(' ', '_')}_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"3D visualizations saved to {output_dir}")
        print(f"GeoJSON data saved to {geojson_path}")
        
        return {
            'composite_map': composite_path,
            'mineral_maps': [os.path.join(output_dir, f"{project_name}_{mineral.replace(' ', '_')}.png") 
                            for mineral in top_minerals],
            'heatmaps': [os.path.join(output_dir, f"{project_name}_{mineral.replace(' ', '_')}_heatmap.png")
                        for mineral in top_minerals],
            'geojson': geojson_path
        }






def run_prediction_pipeline(knowledge_base_path, grid_size=None, region_bounds=None, output_dir="output"):
    """Run the complete mineral prediction pipeline without ArcGIS dependencies"""
    print("Starting advanced mineral prediction pipeline...")
    
    # Create the predictor
    predictor = MineralPredictor()
    
    # Load the knowledge base
    predictor.load_data(knowledge_base_path)
    
    
    # commented part
#     # Build the advanced RAG system
    predictor.build_advanced_rag()
    
    # Train the models with cross-validation
    predictor.train_advanced_models(cross_val=True, n_folds=5)
    
    # Make predictions for a region
    min_lat, max_lat, min_lon, max_lon = region_bounds
    predictions = predictor.predict_region(min_lat, max_lat, min_lon, max_lon, grid_size=grid_size)
    # print("predictions: ", predictions)
    # Save predictions to file
    os.makedirs(output_dir, exist_ok=True)
    prediction_file = os.path.join(output_dir, "mineral_predictions.pkl")
    joblib.dump(predictions, prediction_file)
    
    # Create visualization
    viz_results = predictor.create_3d_map(predictions, output_dir)
    
    # Generate exploration map
    # exploration_results = predictor.generate_exploration_map(predictions, output_dir)
    
    # Save the trained model
    model_dir = os.path.join(output_dir, "models")
    # predictor.save_model(model_dir)
    
    print(f"\nPipeline complete!")
    print(f"- Trained model saved to: {model_dir}")
    print(f"- Predictions saved to: {prediction_file}")
    print(f"- Visualizations saved to: {output_dir}")
    # print(f"- Exploration map saved to: {exploration_results['map']}")
    
    return {
        'model_dir': model_dir,
        'predictions': prediction_file,
        'visualizations': viz_results,
        # 'exploration': exploration_results
    }

if __name__ == "__main__":
    # Example parameters
    knowledge_base_path = "mineral knowledgebase.xlsx"
    region_bounds = (-30, -25, 115, 120)  # Example region in Australia
    output_dir = "./output"
    
    # Run the pipeline
    # results = run_prediction_pipeline(
    #     knowledge_base_path=knowledge_base_path,

    # )
        # Run the pipeline
    results = run_prediction_pipeline(
        knowledge_base_path=knowledge_base_path,
        region_bounds=region_bounds,
        output_dir=output_dir,
        grid_size=20  # Higher resolution grid
    )
    
    print("\nExample of making a prediction for a new location:")
    new_lat, new_lon = -27.5, 117.5  # Example point
    
    # Load the trained model
    predictor = MineralPredictor.load_model(results['model_dir'])
    
    # Make prediction
    prediction = predictor.predict(new_lat, new_lon)
    
    print(f"Predicted minerals at ({new_lat}, {new_lon}):")
    for mineral, prob in prediction['predictions'][:3]:
        print(f"- {mineral}: {prob:.4f} probability")

