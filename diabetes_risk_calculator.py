from diabetes_predictor import DiabetesPredictor
import os

def get_user_input():
    """Get user input for diabetes risk calculation"""
    print("\n=== Diabetes Risk Calculator ===")
    print("Please enter your information below:\n")
    
    try:
        data = {}
        
        # Weight
        while True:
            try:
                weight = float(input("Enter your weight in kg: "))
                if 20 <= weight <= 300:
                    data['BMXWT'] = weight
                    break
                print("Please enter a valid weight between 20 and 300 kg")
            except ValueError:
                print("Please enter a valid number")
        
        # Height
        while True:
            try:
                height = float(input("Enter your height in cm: "))
                if 100 <= height <= 250:
                    data['BMXHT'] = height
                    break
                print("Please enter a valid height between 100 and 250 cm")
            except ValueError:
                print("Please enter a valid number")
        
        # Calculate BMI
        data['BMXBMI'] = weight / ((height/100) ** 2)
        
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
        
        # Waist circumference
        while True:
            try:
                waist = float(input("Enter your waist circumference in cm: "))
                if 50 <= waist <= 200:
                    data['BMXWAIST'] = waist
                    break
                print("Please enter a valid waist circumference between 50 and 200 cm")
            except ValueError:
                print("Please enter a valid number")
        
        # Hip circumference
        while True:
            try:
                hip = float(input("Enter your hip circumference in cm: "))
                if 50 <= hip <= 200:
                    data['BMXHIP'] = hip
                    break
                print("Please enter a valid hip circumference between 50 and 200 cm")
            except ValueError:
                print("Please enter a valid number")
        
        # Upper arm circumference
        while True:
            try:
                arm_circ = float(input("Enter your upper arm circumference in cm: "))
                if 20 <= arm_circ <= 100:
                    data['BMXARMC'] = arm_circ
                    break
                print("Please enter a valid arm circumference between 20 and 100 cm")
            except ValueError:
                print("Please enter a valid number")
        
        # Upper arm length
        while True:
            try:
                arm_length = float(input("Enter your upper arm length in cm: "))
                if 20 <= arm_length <= 100:
                    data['BMXARML'] = arm_length
                    break
                print("Please enter a valid arm length between 20 and 100 cm")
            except ValueError:
                print("Please enter a valid number")
        
        return data
    
    except KeyboardInterrupt:
        print("\nInput cancelled by user")
        return None

def main():
    # Initialize predictor
    predictor = DiabetesPredictor()
    
    # Check if model exists
    if not os.path.exists('diabetes_model.joblib'):
        print("Training new model...")
        # Define file paths
        demo_files = ["DEMO.xpt", "DEMO_B.xpt", "DEMO_C.xpt", "DEMO_D.xpt", "DEMO_E.xpt", "DEMO_F.xpt"]
        diq_files = ["DIQ.xpt", "DIQ_B.xpt", "DIQ_C.xpt", "DIQ_D.xpt", "DIQ_E.xpt", "DIQ_F.xpt"]
        bmx_files = ["BMX.xpt", "BMX_B.xpt", "BMX_C.xpt", "BMX_D.xpt", "BMX_E.xpt", "BMX_F.xpt"]
        
        try:
            # Load and prepare data
            X, y = predictor.load_and_prepare_data(demo_files, diq_files, bmx_files)
            
            # Train model
            predictor.train(X, y)
            
            # Save the model
            predictor.save_model()
            print("Model trained and saved successfully!")
        except Exception as e:
            print(f"Error training model: {str(e)}")
            print("\nPlease make sure all required NHANES data files are present in the current directory.")
            print("Required files:")
            print("- DEMO.xpt files (DEMO.xpt, DEMO_B.xpt, etc.)")
            print("- DIQ.xpt files (DIQ.xpt, DIQ_B.xpt, etc.)")
            print("- BMX.xpt files (BMX.xpt, BMX_B.xpt, etc.)")
            return
    else:
        print("Loading existing model...")
        predictor.load_model()
        print("Model loaded successfully!")
    
    while True:
        # Get user input
        user_data = get_user_input()
        
        if user_data is None:
            break
        
        try:
            # Calculate probability
            probability = predictor.predict_probability(user_data)
            
            # Display results
            print("\n=== Results ===")
            print(f"Your estimated risk of diabetes: {probability:.1%}")
            
            # Risk level interpretation
            if probability < 0.1:
                print("Risk Level: Low")
            elif probability < 0.3:
                print("Risk Level: Moderate")
            else:
                print("Risk Level: High")
            
            print("\nNote: This is an estimate based on demographic and anthropometric data.")
            print("Please consult with a healthcare professional for accurate diagnosis.")
            
        except Exception as e:
            print(f"Error calculating risk: {str(e)}")
            print("Please make sure all required measurements are entered correctly.")
        
        # Ask if user wants to calculate again
        while True:
            again = input("\nWould you like to calculate another risk assessment? (y/n): ").lower()
            if again in ['y', 'n']:
                break
            print("Please enter y or n")
        
        if again == 'n':
            break
    
    print("\nThank you for using the Diabetes Risk Calculator!")

if __name__ == "__main__":
    main() 