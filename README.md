# Income Prediction with Explainable Machine Learning

This project focuses on predicting whether an individual's income exceeds $50K per year using the UCI Adult dataset. It employs two classification models—Random Forest and XGBoost—and applies model explainability techniques including SHAP and LIME to interpret the results. The goal is to build an accurate, interpretable machine learning pipeline suitable for real-world decision-making scenarios.

## Project Overview

The dataset, commonly referred to as the "Adult Income" dataset, contains demographic and employment-related attributes. The task is a binary classification problem where the target variable is whether a person earns more than $50K annually.

The notebook walks through a complete machine learning workflow:
- Data loading and cleaning
- Encoding and feature scaling
- Model training (Random Forest and XGBoost)
- Model evaluation
- Model interpretation using SHAP and LIME

## Dataset

- **Source**: UCI Machine Learning Repository  
- **URL**: https://archive.ics.uci.edu/ml/datasets/adult  
- **Attributes**: Age, education, workclass, occupation, hours-per-week, etc.  
- **Target**: Binary label (`>50K`, `<=50K`)

Missing values are handled by removing incomplete rows, and categorical variables are encoded using `LabelEncoder`.

## Model Training and Evaluation

Two models were trained:
1. **Random Forest Classifier**
2. **XGBoost Classifier**

Feature scaling was applied using `StandardScaler` to improve model performance. The dataset was split into training and test sets (80/20 ratio).

### Evaluation Results

| Model         | Accuracy | F1 Score (High Income) | Recall (High Income) | Precision (High Income) |
|---------------|----------|------------------------|-----------------------|--------------------------|
| Random Forest | 86.09%   | 0.69                   | 0.64                  | 0.75                     |
| XGBoost       | 87.18%   | 0.72                   | 0.67                  | 0.77                     |

XGBoost performs slightly better, particularly in identifying individuals earning more than $50K, which is the minority class. It offers a better balance between precision and recall for high-income predictions.

## Model Explainability

### SHAP (SHapley Additive exPlanations)

SHAP was used to interpret the XGBoost model. It provides global and local explanations by assigning importance values to each feature based on its contribution to the prediction. The summary plot highlights features like education level, capital gain, and hours per week as key contributors.

### LIME (Local Interpretable Model-Agnostic Explanations)

LIME was applied to the Random Forest model to explain individual predictions. It perturbs the input data locally and fits an interpretable model to explain the prediction. This helps understand how specific features influence a single prediction outcome.

## Conclusion

This project demonstrates how to build a reliable income classification model and interpret its decisions using state-of-the-art explainability tools. The inclusion of SHAP and LIME ensures transparency, making the model outputs more trustworthy for stakeholders and users.

## Getting Started

To run this project, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap lime
````

Then, run the Jupyter notebook file:

`Income_Prediction_With_Explainability.ipynb`

## Author

Ahmed Mohammed
GitHub: [https://github.com/ahmdmohamedd](https://github.com/ahmdmohamedd)

```
