# 📊 Bank Customer Churn & Salary Prediction using ANN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-yellow)

A comprehensive Artificial Neural Network (ANN) project that predicts whether a bank customer is likely to churn (leave the bank) and estimates their salary based on various demographic and financial factors. The project features a fully interactive web dashboard built with Streamlit.

## 🌟 Features

- **Churn Prediction (Classification)**: Predicts the probability of a customer leaving the bank using a trained ANN classification model (`model.h5`).
- **Salary Estimation (Regression)**: Estimates a customer's annual salary using a trained ANN regression model (`regression_model.h5`).
- **Interactive Web Dashboard**: A user-friendly Streamlit interface (`visual.py`) allowing users to input customer details and get real-time predictions.
- **Rich Visualizations**: Utilizes Plotly to display gauge charts for salary estimation and pie charts for churn probability.
- **Model Training & Tuning**: Includes Jupyter notebooks for training, prediction, and hyperparameter tuning of the neural networks.

## 📁 Project Structure

All core project files are located within the `myvenv` directory:

```text
📦 Churn modeling using ANN
 ┣ 📂 myvenv
 ┃ ┣ 📜 Churn_Modelling.csv             # Dataset used for training
 ┃ ┣ 📜 training.ipynb                  # ANN classification model training notebook
 ┃ ┣ 📜 regerssion.ipynb                # ANN regression model training notebook
 ┃ ┣ 📜 prediction.ipynb                # Inference and prediction testing
 ┃ ┣ 📜 hyperparamtertunning.ipynb      # Hyperparameter tuning for the models
 ┃ ┣ 📜 visual.py                       # Streamlit web application dashboard
 ┃ ┣ 📜 model.h5                        # Saved classification model
 ┃ ┣ 📜 regression_model.h5             # Saved regression model
 ┃ ┣ 📜 *.pkl                           # Saved scalers and encoders (Scikit-Learn)
 ┃ ┣ 📜 requiremts.txt                  # Python dependencies
 ┃ ┗ 📂 logs                            # TensorBoard logs
```

## 🛠️ Installation & Setup

1. **Clone the repository** (or download the project folder):
   ```bash
   git clone <repository-url>
   cd "Churn modeling using ANN"
   ```

2. **Activate the virtual environment** (Optional but recommended):
   ```bash
   # Windows
   myvenv\Scripts\activate
   # macOS/Linux
   source myvenv/bin/activate
   ```

3. **Install the required dependencies**:
   Navigate to the `myvenv` folder or run from root:
   ```bash
   pip install -r myvenv/requiremts.txt
   ```
   *Note: Ensure you also have `tensorflow` and `plotly` installed if they are not included in the requirements file.*
   ```bash
   pip install tensorflow plotly
   ```

## 🚀 Running the Streamlit Application

To launch the interactive web dashboard, navigate to the `myvenv` directory and run the following Streamlit command:

```bash
cd myvenv
streamlit run visual.py
```

This will start a local web server, and you can view the dashboard in your browser at `http://localhost:8501`.

## 🧠 Model Details

### 1. Classification Model (Churn)
- **Input Features**: Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Estimated Salary.
- **Output**: Probability of churn (0 to 1).
- **Preprocessing**: Label Encoding for Gender, One-Hot Encoding for Geography, Standard Scaling for numerical features.

### 2. Regression Model (Salary)
- **Input Features**: Credit Score, Geography, Gender, Age, Tenure, Balance, Number of Products, Has Credit Card, Is Active Member, Exited (Churn Status).
- **Output**: Estimated Annual Salary (Continuous value).

## 📊 Dataset
The models are trained on the **Bank Customer Churn dataset** (`Churn_Modelling.csv`), which contains 10,000 rows of customer data including their demographics, bank product usage, and churn status.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
