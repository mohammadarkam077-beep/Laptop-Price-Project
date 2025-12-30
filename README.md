💻 Laptop Price Prediction using Machine Learning (VS Code Setup)
📌 Project Overview

This project predicts laptop prices (in Euros) using machine learning based on hardware specifications such as CPU, RAM, storage, GPU, OS, screen quality, and weight.

The project is implemented and executed using Visual Studio Code (VS Code) with Python.

🎯 Problem Statement

To build and compare multiple regression models to accurately predict laptop prices and identify key factors influencing pricing.

📂 Dataset Details

File: laptop_prices.csv

Rows: 1275

Columns: 23

Target Variable: Price_euros

🛠️ Tech Stack

Python 3.9+

VS Code

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn

Jupyter Extension for VS Code

🖥️ Running This Project in VS Code
1️⃣ Install Required Software

Install Python

Install Visual Studio Code

Install VS Code extensions:

Python

Jupyter

2️⃣ Clone the Repository
git clone <your-github-repo-link>
cd laptop-price-prediction

3️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate it:

Windows

venv\Scripts\activate


Mac / Linux

source venv/bin/activate

4️⃣ Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

5️⃣ Open Project in VS Code
code .

6️⃣ Run the Notebook

Open Laptop Price Prediction.ipynb

Select Python interpreter (top-right)

Run cells sequentially

🔍 Exploratory Data Analysis (EDA)

EDA includes:

CPU, GPU, RAM, Storage, OS, Weight analysis

Univariate & Bivariate plots

Correlation heatmaps

Price trend insights

⚙️ Feature Engineering

Created PPI (Pixels Per Inch) from screen resolution

Categorized laptop weights

One-hot encoded categorical variables using ColumnTransformer

🤖 Machine Learning Models Used
Model	R² Score	MAE
KNN Regressor	~0.71	~236
Decision Tree	0.78	226
SVR	❌ Poor	485
Random Forest	0.87	170
Extra Trees	0.886 ⭐	169 ⭐
AdaBoost	0.74	300
🏆 Final Model

Extra Trees Regressor was selected due to:

Highest R² score

Lowest MAE

Better generalization than other models

📈 Evaluation Metrics

R² Score – Model accuracy

Mean Absolute Error (MAE) – Average prediction error in Euros

📁 Project Structure
├── Laptop Price Prediction.ipynb
├── laptop_prices.csv
├── README.md
├── venv/ (optional)
└── saved_model.pkl (optional)

🚀 Future Enhancements

Hyperparameter tuning

Model deployment using Streamlit

Real-time laptop price predictor

Automated model comparison

🧾 Conclusion

This project demonstrates a complete machine learning workflow in VS Code, covering data analysis, feature engineering, model building, evaluation, and final model selection.
Extra Trees Regressor provided the best performance for laptop price prediction.

👤 Author

Md Arkam
Data Analytics & Machine Learning
