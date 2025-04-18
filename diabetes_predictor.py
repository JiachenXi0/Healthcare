import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import os
import pickle
from sklearn.feature_selection import SelectKBest, f_classif

class DiabetesPredictor:
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, 
                                                  min_samples_split=5, class_weight='balanced', 
                                                  random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   random_state=42),
            'CatBoost': CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1,
                                         random_seed=42, verbose=False),
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced',
                                                   random_state=42),
            'SVM': SVC(probability=True, class_weight='balanced', random_state=42)
        }
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = None
        self.best_model = None
        self.best_model_name = None
        
        # Initial feature set - we'll select the most important ones later
        self.feature_names = [
            'RIDAGEYR',  # Age in years
            'RIAGENDR',  # Gender (1=Male, 2=Female)
            'INDFMPIR',  # Family income to poverty ratio
            'RIDRETH1',  # Race/Hispanic origin
            'DMDEDUC2',  # Education level
            'DMDMARTL',  # Marital status
            'RIDEXPRG',  # Pregnancy status
            'DMDCITZN',  # Citizenship status
            'BMXBMI',    # Body Mass Index
            'BMXWAIST',  # Waist Circumference
            'BPXSY1',    # Systolic Blood Pressure
            'BPXDI1',    # Diastolic Blood Pressure
            'LBXGLU',    # Fasting Glucose
            'LBXIN',     # Insulin
            'LBXGH'      # Glycohemoglobin
        ]
        
    def load_and_prepare_data(self, demo_files, diq_files, dbq_files=None):
        """Load and prepare data from NHANES files"""
        try:
            # Load demographic data
            demo_data = pd.concat([pd.read_sas(file) for file in demo_files])
            
            # Load diabetes data
            diq_data = pd.concat([pd.read_sas(file) for file in diq_files])
            
            # Load dietary data if provided
            if dbq_files:
                dbq_data = pd.concat([pd.read_sas(file) for file in dbq_files])
                data = pd.merge(demo_data, diq_data, on='SEQN', how='inner')
                data = pd.merge(data, dbq_data, on='SEQN', how='left')
            else:
                data = pd.merge(demo_data, diq_data, on='SEQN', how='inner')
            
            # Check which features are available in the data
            available_features = [col for col in self.feature_names if col in data.columns]
            missing_features = [col for col in self.feature_names if col not in data.columns]
            
            if missing_features:
                print(f"Warning: The following features are not available in the data: {missing_features}")
                print("Using only available features for prediction.")
            
            # Update feature names to only include available features
            self.feature_names = available_features
            
            if not self.feature_names:
                raise ValueError("No features available in the data. Please check your data files.")
            
            # Prepare features
            X = data[self.feature_names].copy()
            
            # Handle missing values
            X = self.imputer.fit_transform(X)
            
            # Prepare target
            y = data['DIQ010'].map({1: 1, 2: 0, 3: 0, 7: np.nan, 9: np.nan})
            y = y.dropna()
            X = X[y.index]
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            return X, y
            
        except Exception as e:
            print(f"Error preparing data: {str(e)}")
            raise
    
    def select_features(self, X, y, k=10):
        """Select the k most important features"""
        self.feature_selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = self.feature_selector.fit_transform(X, y)
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_feature_names = [self.feature_names[i] for i in selected_indices]
        return X_selected
    
    def train_and_evaluate_models(self, X, y):
        """Train and evaluate all models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Select features
        X_train_selected = self.select_features(X_train, y_train)
        X_test_selected = self.select_features(X_test, y_test)
        
        # Train and evaluate each model
        results = {}
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train_selected, y_train)
            
            # Get predictions
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Store results
            results[name] = {
                'model': model,
                'roc_auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr
            }
            
            # Plot ROC curve
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
            
            # Plot AUC curve
            thresholds = np.linspace(0, 1, 100)
            auc_scores = []
            for threshold in thresholds:
                y_pred = (y_pred_proba >= threshold).astype(int)
                auc_scores.append(roc_auc_score(y_test, y_pred))
            ax2.plot(thresholds, auc_scores, label=f'{name}')
        
        # ROC curve settings
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves for Different Models')
        ax1.legend(loc="lower right")
        
        # AUC curve settings
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('AUC Score')
        ax2.set_title('AUC Scores vs. Classification Threshold')
        ax2.legend(loc="lower right")
        ax2.grid(True)
        
        # Save the plots
        plt.tight_layout()
        plt.savefig('model_performance.png')
        plt.close()
        
        # Select best model
        self.best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        self.best_model = results[self.best_model_name]['model']
        
        return results
    
    def predict_proba(self, X):
        """Predict probability of diabetes using the best model"""
        if self.best_model is None:
            raise ValueError("No model trained. Please train models first.")
        
        # Scale features if needed
        if isinstance(X, pd.DataFrame):
            X = self.scaler.transform(X)
        
        # Select features
        X_selected = self.feature_selector.transform(X)
        
        return self.best_model.predict_proba(X_selected)
    
    def save_model(self):
        """Save the trained model and feature selector"""
        if self.best_model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        with open('diabetes_model.pkl', 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'selected_feature_names': self.selected_feature_names,
                'model_name': self.best_model_name
            }, f)
    
    def load_model(self):
        """Load a trained model"""
        try:
            with open('diabetes_model.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                self.best_model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.feature_selector = saved_data['feature_selector']
                self.selected_feature_names = saved_data['selected_feature_names']
                self.best_model_name = saved_data['model_name']
        except FileNotFoundError:
            raise ValueError("No saved model found. Please train the model first.")

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Define file paths
    demo_files = ["DEMO.xpt", "DEMO_B.xpt", "DEMO_C.xpt", "DEMO_D.xpt", "DEMO_E.xpt", "DEMO_F.xpt"]
    diq_files = ["DIQ.xpt", "DIQ_B.xpt", "DIQ_C.xpt", "DIQ_D.xpt", "DIQ_E.xpt", "DIQ_F.xpt"]
    dbq_files = ["DBQ.xpt", "DBQ_B.xpt", "DBQ_C.xpt", "DBQ_D.xpt", "DBQ_E.xpt", "DBQ_F.xpt"]
    
    # Load and prepare data
    X, y = predictor.load_and_prepare_data(demo_files, diq_files, dbq_files)
    
    # Train and evaluate models
    results = predictor.train_and_evaluate_models(X, y)
    
    # Print results
    print("\nModel Performance (AUC):")
    for name, result in results.items():
        print(f"{name}: {result['roc_auc']:.3f}")
    
    print(f"\nBest performing model: {predictor.best_model_name}")
    print("\nSelected features:")
    for feature in predictor.selected_feature_names:
        print(f"- {feature}")
    
    # Example prediction with only available features
    example_data = {
        'RIDAGEYR': 45,   # Age
        'RIAGENDR': 1,    # Gender (1=Male)
        'INDFMPIR': 2.5,  # Family income to poverty ratio
        'RIDRETH1': 3,    # Race/ethnicity
        'DMDEDUC2': 4,    # Education level
        'DMDMARTL': 1,    # Marital status
        'RIDEXPRG': 2,    # Pregnancy status
        'DMDCITZN': 1     # Citizenship status
    }
    
    # Add additional features if they're available
    if 'BMXBMI' in predictor.feature_names:
        example_data['BMXBMI'] = 28.5
    if 'BMXWAIST' in predictor.feature_names:
        example_data['BMXWAIST'] = 95
    if 'BPXSY1' in predictor.feature_names:
        example_data['BPXSY1'] = 130
    if 'BPXDI1' in predictor.feature_names:
        example_data['BPXDI1'] = 85
    if 'LBXGLU' in predictor.feature_names:
        example_data['LBXGLU'] = 100
    if 'LBXIN' in predictor.feature_names:
        example_data['LBXIN'] = 15
    if 'LBXGH' in predictor.feature_names:
        example_data['LBXGH'] = 5.5
    
    # Convert to DataFrame for prediction
    example_df = pd.DataFrame([example_data])
    probability = predictor.predict_proba(example_df)
    print(f"\nPredicted probability of diabetes: {probability[0][1]:.2%}")
    
    # Save the model
    predictor.save_model() 