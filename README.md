# BigMart-Sales-Prediction
BigMart Sales Prediction
Project Overview
This project aims to predict the sales of various products across BigMart stores using historical data. The goal is to build a regression model that forecasts Item_Outlet_Sales based on product and store attributes, helping optimize inventory and improve sales strategies.

Dataset
Source: Kaggle BigMart Sales Dataset

Size: 8523 records with 12 columns

Features include:

Product info: Item_Identifier, Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP

Store info: Outlet_Identifier, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type

Target variable: Item_Outlet_Sales (continuous sales value)

Data Exploration & Cleaning
Loaded dataset using pandas.

Identified and handled missing data in Item_Weight and Outlet_Size.

Imputed missing weights with the mean, and outlet sizes with the mode.

Visualized distributions of key categorical variables and numeric features.

Feature Engineering & Encoding
Converted categorical variables into numeric codes using Label Encoding for compatibility with sklearn models.

Prepared feature matrix X and target vector y.

Model Development
Split data into training (80%) and test (20%) sets with fixed randomness (random_state=42).

Trained a Random Forest Regressor with 100 trees.

Evaluated model performance on test data.

Model Evaluation
Root Mean Squared Error (RMSE): ~1080.82

R² Score: 0.57 (57% of sales variance explained by the model)

Hyperparameter Tuning and Model Validation
We performed extensive hyperparameter tuning on the XGBoost model with parameters such as learning rate, max depth, feature subsampling, and minimum child weight. The best combination found was:

learning_rate = 0.05

max_depth = 4

n_estimators = 100

colsample_bytree = 0.8

subsample = 1.0

min_child_weight = 1

Using 5-fold cross-validation, our optimized model achieved an average RMSE of 1092.07, with a standard deviation across folds of about 14, indicating consistent performance.

Usage
To run the project locally:

Clone the repository

Install dependencies (e.g., pandas, scikit-learn)

Run the Jupyter notebook or Python scripts for data processing and model training

bash
pip install -r requirements.txt
Next Steps
Extensive feature engineering and hyperparameter tuning

Test alternative models like Gradient Boosting Machines

Implement cross-validation and improve error analysis

Visualize feature importance and residuals

Files Included
BigMart_Sales_Prediction.ipynb: Jupyter notebook with the full workflow

README.md: Project documentation

(Add any other scripts, datasets, or resources)

Acknowledgements
Dataset provided by Kaggle

Inspired by various machine learning tutorials and best practices
