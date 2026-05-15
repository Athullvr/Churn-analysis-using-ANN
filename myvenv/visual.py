import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(page_title="Churn Prediction", layout="wide")

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained classification model
try:
    model_clf = tf.keras.models.load_model(os.path.join(BASE_DIR, 'model.h5'))
except:
    model_clf = None

# Load the trained regression model (if saved)
try:
    model_reg = tf.keras.models.load_model(os.path.join(BASE_DIR, 'regression_model.h5'))
except:
    model_reg = None

# Load the encoders and scalers
with open(os.path.join(BASE_DIR, 'label_encode_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

# Classification scaler
with open(os.path.join(BASE_DIR, 'sclaer.pkl'), 'rb') as file:
    scaler_clf = pickle.load(file)

# Regression scaler (the one saved in regerssion.ipynb is 'scaler.pkl')
try:
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as file:
        scaler_reg = pickle.load(file)
except:
    scaler_reg = None

## streamlit app
st.title('📊 Bank Customer Predictions Dashboard')
st.markdown("---")

tab1, tab2 = st.tabs(["Churn Prediction", "Salary Prediction"])

with tab1:
    if model_clf is None:
        st.error("Classification model not found. Please train and save `model.h5`.")
    else:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.header('Customer Details')
            with st.container(border=True):
                geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='clf_geo')
                gender = st.selectbox('Gender', label_encoder_gender.classes_, key='clf_gen')
                age = st.slider('Age', 18, 92, key='clf_age')
                balance = st.number_input('Balance', key='clf_bal')
                credit_score = st.number_input('Credit Score', key='clf_cr')
                estimated_salary = st.number_input('Estimated Salary', key='clf_sal')
                tenure = st.slider('Tenure', 0, 10, key='clf_tenure')
                num_of_products = st.slider('Number of Products', 1, 4, key='clf_prod')
                
                col_a, col_b = st.columns(2)
                with col_a:
                    has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='clf_card')
                with col_b:
                    is_active_member = st.selectbox('Is Active Member', [0, 1], key='clf_mem')

        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale the input data
        input_data_scaled = scaler_clf.transform(input_data)

        # Predict churn
        prediction = model_clf.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        with col2:
            st.header('Prediction Visuals')
            
            st.subheader(f'Churn Probability: {prediction_proba:.2%}')
            
            # Plotly Pie Chart
            labels = ['Churn', 'Retain']
            values = [prediction_proba, 1 - prediction_proba]
            colors = ['#FF4B4B', '#00CC96'] # Streamlit-like red and green
            
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.5,
                marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
                textinfo='label+percent',
                hoverinfo='label+percent',
                textfont_size=16
            )])
            
            fig.update_layout(
                title_text='Customer Retention vs Churn',
                title_x=0.5,
                annotations=[dict(text='Probability', x=0.5, y=0.5, font_size=20, showarrow=False)],
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Alert banner for the result
            if prediction_proba > 0.5:
                st.error('⚠️ High Risk: The customer is likely to churn.')
            else:
                st.success('✅ Low Risk: The customer is likely to stay.')

with tab2:
    if model_reg is None or scaler_reg is None:
        st.info("💡 **Tip:** To use the Salary Prediction tab, please make sure you save your regression model in `regerssion.ipynb` using `model.save('regression_model.h5')` first!")
    else:
        # Create two columns for layout
        col1_reg, col2_reg = st.columns([1, 1.5])
        
        with col1_reg:
            st.header('Customer Details')
            with st.container(border=True):
                geography_reg = st.selectbox('Geography', onehot_encoder_geo.categories_[0], key='reg_geo')
                gender_reg = st.selectbox('Gender', label_encoder_gender.classes_, key='reg_gen')
                age_reg = st.slider('Age', 18, 92, key='reg_age')
                balance_reg = st.number_input('Balance', key='reg_bal')
                credit_score_reg = st.number_input('Credit Score', key='reg_cr')
                exited_reg = st.selectbox('Customer Exited?', ["No", "Yes"], key='reg_exited')
                tenure_reg = st.slider('Tenure', 0, 10, key='reg_tenure')
                num_of_products_reg = st.slider('Number of Products', 1, 4, key='reg_prod')
                
                col_c, col_d = st.columns(2)
                with col_c:
                    has_cr_card_reg = st.selectbox('Has Credit Card', [0, 1], key='reg_card')
                with col_d:
                    is_active_member_reg = st.selectbox('Is Active Member', [0, 1], key='reg_mem')

        # Prepare the input data (Regression uses Exited but not EstimatedSalary)
        exited_val = 1 if exited_reg == "Yes" else 0
        input_data_reg = pd.DataFrame({
            'CreditScore': [credit_score_reg],
            'Gender': [label_encoder_gender.transform([gender_reg])[0]],
            'Age': [age_reg],
            'Tenure': [tenure_reg],
            'Balance': [balance_reg],
            'NumOfProducts': [num_of_products_reg],
            'HasCrCard': [has_cr_card_reg],
            'IsActiveMember': [is_active_member_reg],
            'Exited': [exited_val]
        })

        # One-hot encode 'Geography'
        geo_encoded_reg = onehot_encoder_geo.transform([[geography_reg]]).toarray()
        geo_encoded_df_reg = pd.DataFrame(geo_encoded_reg, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        # Combine one-hot encoded columns with input data
        input_data_reg = pd.concat([input_data_reg.reset_index(drop=True), geo_encoded_df_reg], axis=1)

        # Scale the input data
        # Note: Regression model used 12 features based on notebook
        input_data_scaled_reg = scaler_reg.transform(input_data_reg)

        # Predict Salary
        salary_pred = model_reg.predict(input_data_scaled_reg)
        predicted_salary = salary_pred[0][0]

        with col2_reg:
            st.header('Predicted Salary')
            
            st.metric(label="Estimated Salary", value=f"${predicted_salary:,.2f}")
            
            # Simple Gauge chart for salary estimation visualization
            fig_reg = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = predicted_salary,
                domain = {'x': [0, 1], 'y': [0, 1]},
                number = {'prefix': "$", 'valueformat': ",.0f"},
                title = {'text': "Estimated Annual Salary"},
                gauge = {
                    'axis': {'range': [None, 250000], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#00CC96"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 50000], 'color': '#ffcccc'},
                        {'range': [50000, 150000], 'color': '#ccffcc'},
                        {'range': [150000, 250000], 'color': '#ccccff'}],
                }
            ))
            
            st.plotly_chart(fig_reg, width='stretch')

