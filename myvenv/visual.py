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

# Load the trained model
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'model.h5'))

# Load the encoders and scaler
with open(os.path.join(BASE_DIR, 'label_encode_gender.pkl'), 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open(os.path.join(BASE_DIR, 'onehot_encoder_geo.pkl'), 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open(os.path.join(BASE_DIR, 'sclaer.pkl'), 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('📊 Customer Churn Prediction Dashboard')
st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.header('Customer Details')
    with st.container(border=True):
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92)
        balance = st.number_input('Balance')
        credit_score = st.number_input('Credit Score')
        estimated_salary = st.number_input('Estimated Salary')
        tenure = st.slider('Tenure', 0, 10)
        num_of_products = st.slider('Number of Products', 1, 4)
        
        col_a, col_b = st.columns(2)
        with col_a:
            has_cr_card = st.selectbox('Has Credit Card', [0, 1])
        with col_b:
            is_active_member = st.selectbox('Is Active Member', [0, 1])

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
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
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
