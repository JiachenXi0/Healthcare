import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
import os

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        # Updated feature names to use only demographic data
        self.feature_names = [
            'RIDAGEYR',  # Age in years
            'RIAGENDR',  # Gender (1=Male, 2=Female)
            'INDFMPIR',  # Family income to poverty ratio
            'RIDRETH1',  # Race/ethnicity
            'DMDEDUC2',  # Education level
            'DMDMARTL',  # Marital status
            'RIDEXPRG',  # Pregnancy status
            'RIDEXAGM',  # Age in months at exam
            'DMDBORN4',  # Country of birth
            'DMDCITZN'   # Citizenship status
        ]
        
    def load_and_prepare_data(self, demo_files, diq_files):
        """Load and prepare the NHANES data"""
        # Load demographic and diabetes data
        demo_data = pd.concat([pd.read_sas(file) for file in demo_files])
        diq_data = pd.concat([pd.read_sas(file) for file in diq_files])
        
        # Print available columns for debugging
        print("Available columns in demo_data:", demo_data.columns.tolist())
        
        # Merge datasets
        merged_data = demo_data.merge(diq_data[['SEQN', 'DIQ010']], on='SEQN', how='inner')
        
        # Filter for diabetes status (1=Yes, 2=No)
        merged_data = merged_data[merged_data['DIQ010'].isin([1, 2])]
        merged_data['DIQ010'] = merged_data['DIQ010'].map({1: 1, 2: 0})
        
        # Select features
        X = merged_data[self.feature_names]
        y = merged_data['DIQ010']
        
        return X, y
    
    def train(self, X, y):
        """Train the diabetes prediction model"""
        # Handle missing values
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Calculate and print model performance
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        return self.model
    
    def predict_probability(self, input_data):
        """Predict probability of diabetes for new data"""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data], columns=self.feature_names)
        
        # Handle missing values and scale
        input_imputed = self.imputer.transform(input_df)
        input_scaled = self.scaler.transform(input_imputed)
        
        # Get probability prediction
        probability = self.model.predict_proba(input_scaled)[0][1]
        return probability
    
    def save_model(self, path='diabetes_model.joblib'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path='diabetes_model.joblib'):
        """Load a trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.feature_names = model_data['feature_names']

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Define file paths
    demo_files = ["DEMO.xpt", "DEMO_B.xpt", "DEMO_C.xpt", "DEMO_D.xpt", "DEMO_E.xpt", "DEMO_F.xpt"]
    diq_files = ["DIQ.xpt", "DIQ_B.xpt", "DIQ_C.xpt", "DIQ_D.xpt", "DIQ_E.xpt", "DIQ_F.xpt"]
    
    # Load and prepare data
    X, y = predictor.load_and_prepare_data(demo_files, diq_files)
    
    # Train model
    predictor.train(X, y)
    
    # Example prediction
    example_data = {
        'RIDAGEYR': 45,   # Age
        'RIAGENDR': 1,    # Gender (1=Male)
        'INDFMPIR': 2.5,  # Family income to poverty ratio
        'RIDRETH1': 3,    # Race/ethnicity
        'DMDEDUC2': 4,    # Education level
        'DMDMARTL': 1,    # Marital status
        'RIDEXPRG': 2,    # Pregnancy status
        'RIDEXAGM': 540,  # Age in months at exam
        'DMDBORN4': 1,    # Country of birth
        'DMDCITZN': 1     # Citizenship status
    }
    
    probability = predictor.predict_probability(example_data)
    print(f"\nPredicted probability of diabetes: {probability:.2%}")
    
    # Save the model
    predictor.save_model() 