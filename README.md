# Diabetes Risk Calculator

This application uses machine learning to estimate an individual's risk of diabetes based on demographic and anthropometric data. The model is trained on NHANES (National Health and Nutrition Examination Survey) data.

## Features

- Calculates diabetes risk probability based on 10 key measurements
- User-friendly command-line interface
- Input validation for all measurements
- Risk level interpretation (Low, Moderate, High)
- Saves trained model for future use

## Requirements

- Python 3.7 or higher
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone this repository or download the files
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the calculator:
   ```bash
   python diabetes_risk_calculator.py
   ```

2. Enter your measurements when prompted:
   - Weight (kg)
   - Height (cm)
   - Age (years)
   - Gender (M/F)
   - Family income to poverty ratio (0-5)
   - Waist circumference (cm)
   - Hip circumference (cm)
   - Upper arm circumference (cm)
   - Upper arm length (cm)

3. View your results:
   - Estimated risk percentage
   - Risk level interpretation
   - Important notes

## Important Notes

- This is a screening tool and should not be used for self-diagnosis
- Always consult with a healthcare professional for accurate diagnosis
- The model is based on demographic and anthropometric data only
- Results are estimates and should be interpreted with caution

## Data Requirements

The model requires NHANES data files (DEMO.xpt and DIQ.xpt) for initial training. These files should be in the same directory as the script.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.