from diabetes_predictor import DiabetesPredictor
import os
import pandas as pd

def get_user_input():
    """Get user input for diabetes risk calculation"""
    print("\n=== Diabetes Risk Calculator ===")
    print("Please enter your information below:\n")
    
    try:
        data = {}
        
        # Age
        while True:
            try:
                age = int(input("Enter your age in years: "))
                if 0 <= age <= 120:
                    data['RIDAGEYR'] = age
                    break
                print("Please enter a valid age between 0 and 120 years")
            except ValueError:
                print("Please enter a valid number")
        
        # Gender
        while True:
            gender = input("Enter your gender (M/F): ").upper()
            if gender in ['M', 'F']:
                data['RIAGENDR'] = 1 if gender == 'M' else 2
                break
            print("Please enter M or F")
        
        # Family income to poverty ratio
        while True:
            try:
                income_ratio = float(input("Enter your family income to poverty ratio (0-5): "))
                if 0 <= income_ratio <= 5:
                    data['INDFMPIR'] = income_ratio
                    break
                print("Please enter a valid ratio between 0 and 5")
            except ValueError:
                print("Please enter a valid number")
        
        # Race/Hispanic origin
        print("\nRace/Hispanic Origin options:")
        print("1. Mexican American")
        print("2. Other Hispanic")
        print("3. Non-Hispanic White")
        print("4. Non-Hispanic Black")
        print("5. Other Race - Including Multi-Racial")
        while True:
            try:
                race = int(input("Enter your race/Hispanic origin (1-5): "))
                if 1 <= race <= 5:
                    data['RIDRETH1'] = race
                    break
                print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
        
        # Education Level
        print("\nEducation Level options:")
        print("1. Less than 9th grade")
        print("2. 9-11th grade")
        print("3. High school graduate/GED")
        print("4. Some college or AA degree")
        print("5. College graduate or above")
        while True:
            try:
                education = int(input("Enter your education level (1-5): "))
                if 1 <= education <= 5:
                    data['DMDEDUC2'] = education
                    break
                print("Please enter a number between 1 and 5")
            except ValueError:
                print("Please enter a valid number")
        
        # Marital Status
        print("\nMarital Status options:")
        print("1. Married")
        print("2. Widowed")
        print("3. Divorced")
        print("4. Separated")
        print("5. Never married")
        print("6. Living with partner")
        while True:
            try:
                marital = int(input("Enter your marital status (1-6): "))
                if 1 <= marital <= 6:
                    data['DMDMARTL'] = marital
                    break
                print("Please enter a number between 1 and 6")
            except ValueError:
                print("Please enter a valid number")
        
        # Pregnancy Status (for females only)
        if data['RIAGENDR'] == 2:  # Female
            while True:
                pregnant = input("Are you pregnant? (y/n): ").lower()
                if pregnant in ['y', 'n']:
                    data['RIDEXPRG'] = 1 if pregnant == 'y' else 2
                    break
                print("Please enter y or n")
        else:
            data['RIDEXPRG'] = 2  # Not applicable for males
        
        # Citizenship Status
        print("\nCitizenship Status options:")
        print("1. Citizen by birth or naturalization")
        print("2. Not a citizen of the United States")
        while True:
            try:
                citizen = int(input("Enter your citizenship status (1-2): "))
                if 1 <= citizen <= 2:
                    data['DMDCITZN'] = citizen
                    break
                print("Please enter a number between 1 and 2")
            except ValueError:
                print("Please enter a valid number")
        
        return data
    
    except KeyboardInterrupt:
        print("\nInput cancelled by user")
        return None

def calculate_max_risk(predictor):
    """Calculate and display the maximum risk profile probability"""
    max_risk_profile = {
        'RIDAGEYR': 120,      # Maximum age
        'RIAGENDR': 2,        # Female
        'INDFMPIR': 0,        # Lowest income to poverty ratio
        'RIDRETH1': 4,        # Non-Hispanic Black
        'DMDEDUC2': 1,        # Less than 9th grade
        'DMDMARTL': 3,        # Divorced
        'RIDEXPRG': 1,        # Pregnant
        'DMDCITZN': 2         # Not a citizen
    }
    
    # Convert to DataFrame
    max_risk_df = pd.DataFrame([max_risk_profile])
    
    # Calculate probability
    probability = predictor.predict_proba(max_risk_df)[0][1] * 100
    
    print("\n=== Maximum Risk Profile Analysis ===")
    print("\nProfile Characteristics:")
    print(f"Age: {max_risk_profile['RIDAGEYR']} years")
    print("Gender: Female")
    print(f"Income to Poverty Ratio: {max_risk_profile['INDFMPIR']}")
    print("Race/Ethnicity: Non-Hispanic Black")
    print("Education: Less than 9th grade")
    print("Marital Status: Divorced")
    print("Pregnancy Status: Pregnant")
    print("Citizenship: Not a citizen of the United States")
    print(f"\nPredicted Diabetes Risk: {probability:.2f}%")

def main():
    try:
        # Initialize predictor
        predictor = DiabetesPredictor()
        
        # Check if model file exists
        if os.path.exists('diabetes_model.pkl'):
            print("Loading existing model...")
            predictor.load_model()
        else:
            print("Training new model...")
            # Load and prepare data
            demo_files = ['DEMO.xpt', 'DEMO_B.xpt', 'DEMO_C.xpt', 'DEMO_D.xpt', 'DEMO_E.xpt', 'DEMO_F.xpt', 'DEMO_I.xpt']
            diq_files = ['DIQ.xpt', 'DIQ_B.xpt', 'DIQ_C.xpt', 'DIQ_D.xpt', 'DIQ_E.xpt', 'DIQ_F.xpt', 'DIQ_I.xpt']
            
            # Check if required files exist
            missing_files = []
            for file in demo_files + diq_files:
                if not os.path.exists(file):
                    missing_files.append(file)
            
            if missing_files:
                print("\nError: The following required NHANES data files are missing:")
                for file in missing_files:
                    print(f"- {file}")
                print("\nPlease make sure all required NHANES data files are present in the current directory.")
                return
            
            X, y = predictor.load_and_prepare_data(demo_files, diq_files)
            predictor.train_model(X, y)
            predictor.save_model()
        
        # Calculate and display maximum risk profile
        calculate_max_risk(predictor)
        
        # Get user input and calculate risk
        while True:
            user_data = get_user_input()
            if user_data is None:
                break
            
            # Convert to DataFrame
            user_df = pd.DataFrame([user_data])
            
            # Calculate probability
            probability = predictor.predict_proba(user_df)[0][1] * 100
            
            print(f"\nYour predicted diabetes risk: {probability:.2f}%")
            
            # Ask if user wants to calculate another risk
            while True:
                again = input("\nWould you like to calculate another risk? (y/n): ").lower()
                if again in ['y', 'n']:
                    break
                print("Please enter y or n")
            
            if again == 'n':
                break
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease make sure all required NHANES data files are present in the current directory.")
        print("Required files:")
        print("- DEMO.xpt files (DEMO.xpt, DEMO_B.xpt, etc.)")
        print("- DIQ.xpt files (DIQ.xpt, DIQ_B.xpt, etc.)")

if __name__ == "__main__":
    main() 