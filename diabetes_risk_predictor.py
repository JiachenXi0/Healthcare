import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

class DiabetesRiskPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()
        
        # Field descriptions
        self.field_descriptions = {
            'RIDAGEYR': {
                'description': 'Age in Years',
                'input_type': 'numeric',
                'range': '0-120',
                'example': '45'
            },
            'RIAGENDR': {
                'description': 'Gender',
                'input_type': 'categorical',
                'options': {
                    '1': 'Male',
                    '2': 'Female'
                }
            },
            'INDFMPIR': {
                'description': 'Family Income to Poverty Ratio',
                'input_type': 'numeric',
                'range': '0-5',
                'explanation': '0 = Below poverty, 1 = At poverty, >1 = Above poverty',
                'example': '2.5'
            },
            'RIDRETH1': {
                'description': 'Race/Hispanic Origin',
                'input_type': 'categorical',
                'options': {
                    '1': 'Mexican American',
                    '2': 'Other Hispanic',
                    '3': 'Non-Hispanic White',
                    '4': 'Non-Hispanic Black',
                    '5': 'Other Race'
                }
            },
            'DMDEDUC2': {
                'description': 'Education Level',
                'input_type': 'categorical',
                'options': {
                    '1': 'Less than 9th grade',
                    '2': '9-11th grade',
                    '3': 'High school graduate',
                    '4': 'Some college or AA degree',
                    '5': 'College graduate or above'
                }
            },
            'DMDMARTL': {
                'description': 'Marital Status',
                'input_type': 'categorical',
                'options': {
                    '1': 'Married',
                    '2': 'Widowed',
                    '3': 'Divorced',
                    '4': 'Separated',
                    '5': 'Never married',
                    '6': 'Living with partner'
                }
            },
            'RIDEXPRG': {
                'description': 'Pregnancy Status',
                'input_type': 'categorical',
                'options': {
                    '1': 'Yes',
                    '2': 'No',
                    '3': 'Not applicable'
                }
            },
            'DMDCITZN': {
                'description': 'Citizenship Status',
                'input_type': 'categorical',
                'options': {
                    '1': 'Citizen by birth',
                    '2': 'Citizen by naturalization',
                    '3': 'Not a citizen'
                }
            },
            'BMXBMI': {
                'description': 'Body Mass Index',
                'input_type': 'numeric',
                'range': '15-50',
                'explanation': '<18.5 = Underweight, 18.5-24.9 = Normal, 25-29.9 = Overweight, ≥30 = Obese',
                'example': '28.5'
            },
            'BMXWAIST': {
                'description': 'Waist Circumference (cm)',
                'input_type': 'numeric',
                'range': '50-150',
                'example': '95'
            },
            'BPXSY1': {
                'description': 'Systolic Blood Pressure (mmHg)',
                'input_type': 'numeric',
                'range': '90-200',
                'example': '130'
            },
            'BPXDI1': {
                'description': 'Diastolic Blood Pressure (mmHg)',
                'input_type': 'numeric',
                'range': '50-120',
                'example': '85'
            },
            'LBXGLU': {
                'description': 'Fasting Glucose (mg/dL)',
                'input_type': 'numeric',
                'range': '50-300',
                'example': '100'
            },
            'LBXIN': {
                'description': 'Insulin Level (μU/mL)',
                'input_type': 'numeric',
                'range': '2-50',
                'example': '15'
            },
            'LBXGH': {
                'description': 'Glycohemoglobin (HbA1c) (%)',
                'input_type': 'numeric',
                'range': '4-15',
                'example': '5.5'
            }
        }
        
    def load_model(self):
        """Load the trained CatBoost model and scaler"""
        try:
            with open('diabetes_model.pkl', 'rb') as f:
                saved_data = pickle.load(f)
                self.model = saved_data['model']
                self.scaler = saved_data['scaler']
                self.feature_names = saved_data['selected_feature_names']
        except FileNotFoundError:
            raise ValueError("No saved model found. Please train the model first using diabetes_predictor.py")
    
    def show_field_description(self, feature):
        """Display description and expected input for a feature"""
        desc = self.field_descriptions[feature]
        print(f"\n{desc['description']} ({feature})")
        
        if desc['input_type'] == 'categorical':
            print("Options:")
            for key, value in desc['options'].items():
                print(f"  {key} = {value}")
        else:
            print(f"Expected range: {desc['range']}")
            if 'explanation' in desc:
                print(f"Explanation: {desc['explanation']}")
            print(f"Example: {desc['example']}")
    
    def get_feature_input(self):
        """Get user input for each feature"""
        print("\nPlease enter the following information:")
        feature_values = {}
        
        for feature in self.feature_names:
            while True:
                try:
                    # Show field description
                    if feature not in self.field_descriptions:
                        print(f"Error: Unknown field {feature}. Skipping...")
                        break
                        
                    self.show_field_description(feature)
                    
                    # Get input and strip any whitespace
                    value = input(f"Enter value for {feature}: ").strip()
                    
                    # Skip if the input looks like a command or path
                    if any(char in value for char in ['/', '\\', '.py', 'python']):
                        print("Invalid input. Please enter a valid value for the field.")
                        continue
                    
                    # Handle help command
                    if value.lower() == 'help':
                        self.show_help()
                        continue
                    
                    # Validate input
                    if self.field_descriptions[feature]['input_type'] == 'categorical':
                        valid_options = list(self.field_descriptions[feature]['options'].keys())
                        if value not in valid_options:
                            print(f"Please enter one of: {', '.join(valid_options)}")
                            continue
                        feature_values[feature] = float(value)
                        break
                    else:
                        try:
                            value = float(value)
                            min_val, max_val = map(float, self.field_descriptions[feature]['range'].split('-'))
                            if not (min_val <= value <= max_val):
                                print(f"Value must be between {min_val} and {max_val}")
                                continue
                            feature_values[feature] = value
                            break
                        except ValueError:
                            print(f"Please enter a valid number between {min_val} and {max_val}")
                            continue
                    
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    print("Please try again or type 'help' for assistance.")
                    continue
        
        return feature_values
    
    def predict_risk(self, feature_values):
        """Predict diabetes risk based on input features"""
        # Convert to DataFrame
        input_data = pd.DataFrame([feature_values])
        
        # Scale features
        scaled_data = self.scaler.transform(input_data)
        
        # Make prediction
        probability = self.model.predict_proba(scaled_data)[0][1]
        
        return probability * 100  # Convert to percentage
    
    def show_help(self):
        """Display help information"""
        print("\n=== Diabetes Risk Predictor Help ===")
        print("This tool predicts your risk of developing diabetes based on various health and demographic factors.")
        print("\nRisk Level Interpretation:")
        print("- < 10%: Low risk")
        print("- 10-30%: Moderate risk")
        print("- 30-50%: High risk")
        print("- > 50%: Very high risk")
        print("\nFor each field, you'll be shown:")
        print("- Description of the measurement")
        print("- Expected range or valid options")
        print("- Example value")
        print("\nType 'help' at any time to see this information again.")
    
    def run_interactive_prediction(self):
        """Run interactive prediction session"""
        print("\n=== Diabetes Risk Predictor ===")
        self.show_help()
        
        while True:
            try:
                # Get user input
                feature_values = self.get_feature_input()
                
                # Make prediction
                risk_percentage = self.predict_risk(feature_values)
                
                # Display result
                print(f"\nPredicted Diabetes Risk: {risk_percentage:.1f}%")
                
                # Risk interpretation
                if risk_percentage < 10:
                    print("Risk Level: Low")
                elif risk_percentage < 30:
                    print("Risk Level: Moderate")
                elif risk_percentage < 50:
                    print("Risk Level: High")
                else:
                    print("Risk Level: Very High")
                
                # Ask if user wants to make another prediction
                another = input("\nWould you like to make another prediction? (yes/no): ").lower()
                if another != 'yes':
                    break
                    
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                another = input("\nWould you like to try again? (yes/no): ").lower()
                if another != 'yes':
                    break

if __name__ == "__main__":
    try:
        predictor = DiabetesRiskPredictor()
        predictor.run_interactive_prediction()
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please make sure you have trained the model first using diabetes_predictor.py") 