# 🧠 ARC Master GUI

## Overview
The ARC Master GUI is an interactive Streamlit-based application designed to visualize and evaluate machine learning and rule-based models solving Abstract Reasoning Challenge (ARC) tasks. The tool allows users to load ARC datasets, visualize data insights, apply reasoning models, and visually inspect predictions alongside ground truths.

## Key Features
- **Dataset Loading:** Load ARC datasets (Training, Evaluation, Test) either from local paths or directly via file uploads.
- **Visualization:**
  - Grid Size Distribution: Visualizes the frequency of various grid dimensions.
  - Color Frequency: Displays how often different colors appear within the datasets.
  - Grid Visualization: Renders ARC grids visually with clear color mapping.
- **Reasoning Models:**
  - Rule-Based Model (simple transformations like horizontal flips)
  - Extendable to include advanced ML models like XGBoost or SVM.
- **Interactive Predictions:** Displays test inputs, model-predicted outputs, and actual ground truths for straightforward visual comparison.
- **Explanations:** Provides clear textual explanations for each model prediction.

## Installation
```bash
pip install streamlit numpy matplotlib xgboost scikit-learn
```

## Usage
- Run the app using Streamlit:
  ```bash
  streamlit run your_script.py
  ```
- Select dataset type (Training/Evaluation/Test) and load the data.
- Choose task ID and reasoning model to see predictions and explanations.

## File Structure
- `your_script.py`: Main Streamlit application.
- Dataset files (`.json`) should be organized clearly for easy loading.

## Requirements
- Python >= 3.7
- Streamlit
- NumPy
- Matplotlib
- XGBoost
- Scikit-learn

## Author
Developed by Ngoubi Maximillian Diangha .

## License
MIT License

