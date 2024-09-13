import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the model and encoders
model = load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Custom CSS and Styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f0f2f6;
        }
        h1 {
            color: purple;
            text-align: center;
        }
        .report-container {
            background-color: #ffffff;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
        }
        .prediction-text {
            font-size: 20px;
            text-align: center;
            color: #333333;
        }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app title
st.markdown("<h1>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("---")  # Separator line

# Input fields using columns
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, 30)
    balance = st.number_input('Balance', value=0.0, step=1000.0)

with col2:
    credit_score = st.slider('Credit Score', 300, 850, 600)
    estimated_salary = st.number_input('Estimated Salary', value=0.0, step=1000.0)
    tenure = st.slider('Tenure', 0, 10, 5)
    num_of_products = st.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('Has Credit Card', [0, 1])
    is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prediction function
def predict_churn(input_data):
    # One-hot encode geography
    geo_encoded = one_hot_encoder_geo.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
    
    # Prepare input dataframe
    input_df = pd.DataFrame({
        'CreditScore': [input_data['CreditScore']],
        'Gender': [label_encoder_gender.transform([input_data['Gender']])[0]],
        'Age': [input_data['Age']],
        'Tenure': [input_data['Tenure']],
        'Balance': [input_data['Balance']],
        'NumOfProducts': [input_data['NumOfProducts']],
        'HasCrCard': [input_data['HasCrCard']],
        'IsActiveMember': [input_data['IsActiveMember']],
        'EstimatedSalary': [input_data['EstimatedSalary']]
    })
    
    input_df = pd.concat([input_df, geo_encoded_df], axis=1)
    
    # Scale the input data
    input_scaled = scaler.transform(input_df)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    return prediction[0][0]

## Predict button
if st.button('Predict Churn'):
    input_data = {
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Balance': balance,
        'CreditScore': credit_score,
        'EstimatedSalary': estimated_salary,
        'Tenure': tenure,
        'NumOfProducts': num_of_products,
        'HasCrCard': has_cr_card,
        'IsActiveMember': is_active_member
    }
    
    churn_probability = predict_churn(input_data)
    
    # Display the churn probability as a progress bar and text
    st.markdown("<div class='prediction-text'>Churn Probability:</div>", unsafe_allow_html=True)
    st.progress(int(churn_probability * 100))  # Scale and convert to int for progress bar
    
    # Display prediction outcome with color indicators
    if churn_probability > 0.5:
        st.markdown(f"<h3 style='color: red; text-align: center;'>⚠️ Likely to Churn ({churn_probability:.2%})</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: green; text-align: center;'>✔️ Likely to Stay ({churn_probability:.2%})</h3>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center;'>Customer Profile</h4>", unsafe_allow_html=True)
    st.json(input_data)

    # Plot a simple bar chart for the customer’s financial details
    fig, ax = plt.subplots()
    ax.bar(['Credit Score', 'Balance', 'Salary'], [credit_score, balance, estimated_salary], color=['purple', 'blue', 'green'])
    st.pyplot(fig)

    # Download customer data as CSV
    csv_data = pd.DataFrame(input_data, index=[0]).to_csv(index=False)
    st.download_button(label="Download Customer Data as CSV", data=csv_data, file_name='customer_data.csv', mime='text/csv')

# Model information and guidance
st.write("""
### About this model
This prediction is based on a neural network model trained on historical customer data. 
The model considers various factors such as credit score, geography, gender, age, and account details to estimate the likelihood of a customer leaving the bank.

### How to interpret the results
- A probability below 50% suggests the customer is likely to stay.
- A probability above 50% indicates a risk of churning.

### Tips for retention
- For high-risk customers, consider personalized retention offers or improved services.
- For low-risk customers, maintain the quality of service and look for opportunities to increase their engagement with the bank.
""")
