# Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease based on patient health and clinical measurements.

The project uses the UCI Cleveland Heart Disease dataset and compares several classification algorithms to find a model that performs well on the available data.

## Project Overview

The goal is to predict the `target` value for a patient using 13 input features.

- `1` - presence of heart disease / associated risk factors
- `0` - absence of heart disease / associated risk factors

This project focuses on the machine learning workflow rather than clinical diagnosis.

## Dataset

The dataset used in this project is based on the UCI / Cleveland Heart Disease dataset.

The cleaned dataset contains **302 unique patient records**, with 13 input features and 1 target column.

One important issue with this dataset is that several publicly available versions contain 1,025 rows. A large portion of those rows are duplicates of the same patient records. Using the duplicated version can cause the same patients to appear in both the training and test sets, resulting in unrealistically high accuracy.

For this reason, the dataset used here was deduplicated before training and evaluation.

### Features

| Feature | Description |
|---|---|
| `age` | Age of the patient in years |
| `sex` | Sex of the patient |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression caused by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels observed by fluoroscopy |
| `thal` | Thalassemia indicator |
| `target` | Prediction target |

## Machine Learning Approach

The project follows these main steps:

1. **Data cleaning**
   - Removed duplicate patient records.
   - Checked the dataset for missing values and inconsistencies.

2. **Exploratory Data Analysis**
   - Examined the distribution of the target variable.
   - Created a correlation heatmap.
   - Compared individual features against the target.
   - Looked at age distributions for different outcomes.

3. **Model comparison**

   Several classification algorithms were tested:

   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
   - AdaBoost
   - K-Nearest Neighbors (KNN)

   Models were compared using **5-fold cross-validation** rather than relying on a single train/test split.

4. **Hyperparameter tuning**

   `GridSearchCV` was used to tune the better-performing model and find a more suitable set of parameters.

5. **Model evaluation**

   The final model was evaluated using:

   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix

6. **Feature importance**

   Feature importance was also examined to understand which input variables contributed most to the model's predictions.

7. **Model export**

   The final trained model was saved using `joblib` so that it can be loaded and reused without retraining.

## Results

The initial model comparison produced the following approximate cross-validation accuracy:

| Model | Cross-Validation Accuracy |
|---|---:|
| Logistic Regression | ~84% |
| SVM (Linear) | ~83% |
| Random Forest | ~82% |
| AdaBoost | ~82% |
| KNN (k=5) | ~65% |

Random Forest was selected for further tuning.

The tuned model provides more realistic results than the 100% accuracy that can be obtained from some duplicated versions of the dataset.

The notebook contains the detailed evaluation results, including precision, recall, F1-score and the confusion matrix.

## Project Files

- `heart.csv` — cleaned, deduplicated dataset (302 rows)
- `heart_disease_prediction.ipynb` — full analysis notebook (EDA, cross-validation, tuning, evaluation, feature importance)
- `heart_disease_model.joblib` — final trained model, ready to load and reuse
- `requirements.txt` — Python dependencies
  
## Usage

1. **Clone the repository**

```bash
git clone https://github.com/FerdousRIAZzz/heart_disease_prediction.git
cd heart-disease-prediction
```

2. **Install the dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the notebook**

```bash
jupyter notebook heart_disease_prediction.ipynb
```

You can then run the notebook cells to reproduce the analysis and model training process.

The saved model can also be loaded directly with Python:

```python

import joblib
import pandas as pd

model = joblib.load("heart_disease_model.joblib")

patient = pd.DataFrame([{
    "age": 59,
    "sex": 1,
    "cp": 0,
    "trestbps": 140,
    "chol": 221,
    "fbs": 0,
    "restecg": 1,
    "thalach": 164,
    "exang": 1,
    "oldpeak": 0.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
}])

prediction = model.predict(patient)[0]

probability = model.predict_proba(patient)[0][1]

print("Prediction:", prediction)
print("Probability:", probability)
```

## Next Steps

- Compare against gradient boosting models (XGBoost, LightGBM) with the same cross-validation protocol.
- Validate on a larger, independently-sourced dataset to test generalization beyond these 302 patients.
- Build a small interactive demo (e.g. Streamlit) around the saved model.

## Disclaimer

This project was created for educational and machine learning practice purposes.
It is not a medical diagnostic tool and should not be used to make medical or clinical decisions.
