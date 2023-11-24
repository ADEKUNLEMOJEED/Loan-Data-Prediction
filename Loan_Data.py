import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('Loan_prediction_dataset.csv')
df = data.copy()

# df['Dependents'].sample(200).values
df['Dependents'] = df['Dependents'].str.replace('+', '')
df['Dependents'] = df['Dependents'].astype(float)
# df['Dependents'].dtypes

def cleaner(dataframe):
    for i in dataframe.columns:
        if (dataframe[i].isnull().sum()/len(dataframe) * 100) > 30:
            dataframe.drop(i, inplace = True, axis = 1)

        elif dataframe[i].dtypes != 'O':
            dataframe[i].fillna(dataframe[i].median(), inplace = True)

        else:
            dataframe[i].fillna(dataframe[i].mode()[0], inplace = True)
cleaner(df)

sel = df.copy()

categoricals = df.select_dtypes(include = ['object', 'category'])
numericals = df.select_dtypes(include = 'number')

# PREPROCESSSING
# Standardization
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
encoder = LabelEncoder()

for i in numericals.columns: # ................................................. Select all numerical columns
    if i in df.columns: # ...................................................... If the selected column is found in the general dataframe
        df[i] = scaler.fit_transform(df[[i]]) # ................................ Scale it

for i in categoricals.columns: # ............................................... Select all categorical columns
    if i in df.columns: # ...................................................... If the selected columns are found in the general dataframe
        df[i] = encoder.fit_transform(df[i])# .................................. encode it

df.drop('Loan_ID', axis = 1, inplace = True)
selected_columns = ['ApplicantIncome', 'LoanAmount', 'CoapplicantIncome', 'Dependents', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']
dx = df[selected_columns]

x = dx
y = df.Loan_Status

# split into train and test
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.10, random_state= 43, stratify = y)
# Modelling
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression()
model.fit(xtrain, ytrain)
cross_validation = model.predict(xtrain)
pred = model.predict(xtest)


# save model
model = pickle.dump(model, open('Loan_Data.pkl', 'wb'))

# ..............STREAMLIT DEVELOPEMENT..........
model = pickle.load(open('Loan_Data.pkl','rb'))

# Streamlit app header

st.markdown("<h1 style = 'color: #1F1717; text-align: center;font-family: Arial, Helvetica, sans-serif; '>LOAN PREDICTION</h1>", unsafe_allow_html= True)
st.markdown('<br2>', unsafe_allow_html= True)
st.markdown("<h5 style = 'margin: -25px; color: #D36B00; text-align: center;font-family: Arial, Helvetica, sans-serif; '> Built By Adekunle Mojeed</h5>", unsafe_allow_html= True)
st.sidebar.markdown('<br><br><br>', unsafe_allow_html= True)

# Sidebar navigation

st.sidebar.image('pngwing.com (18).png', width = 200)
selected_page = st.sidebar.radio("Navigation", ["Home", "Prediction"])

# Function to define the home page content
def home_page():
    st.image('pngwing.com (17).png', width = 600)
    st.markdown("<h2 style='color: #990000;'>Project Background</h2>", unsafe_allow_html=True)
    st.write("""
   The Loan Prediction App aims to revolutionize this landscape by leveraging advanced machine learning algorithms to streamline and enhance the loan approval process. By harnessing the power of predictive analytics, this app provides a swift and reliable assessment of the likelihood of loan approval, contributing to a more efficient and customer-friendly lending ecosystem..
    """)
    st.sidebar.markdown('<br>', unsafe_allow_html= True)



# Function to define the loan prediction page content
def prediction_page():
    st.markdown("<div style='text-align: center;'><img src='pngwing.com (17).png' alt='Loan Prediction App' width='300'></div>", unsafe_allow_html=True)

    st.sidebar.markdown("Add your modeling content here")
    st.write(sel.head())
    st.sidebar.image('pngwing.com (20).png', width = 300)
     # Collect user input
    applicant_income = st.sidebar.slider("Applicant Income", sel['ApplicantIncome'].min(), sel['ApplicantIncome'].max())
    loan_amount = st.sidebar.slider("Loan Amount", sel['LoanAmount'].min(), sel['LoanAmount'].max())
    coapplicant_income = st.sidebar.slider("Coapplicant Income", sel['CoapplicantIncome'].min(), sel['CoapplicantIncome'].max())
    dependents = st.sidebar.slider("Dependents", sel['Dependents'].min(), sel['Dependents'].max())
    property_area = st.sidebar.selectbox("Property Area", sel['Property_Area'].unique())
    credit_history = st.sidebar.slider("Credit History", sel['Credit_History'].min(), sel['Credit_History'].max())
    loan_amount_term = st.sidebar.slider("Loan Amount Term", sel['Loan_Amount_Term'].min(), sel['Loan_Amount_Term'].max())

    user_input = {} # Initialize user_input
    # Create a dictionary from user input
    user_input = {
        'ApplicantIncome': applicant_income,
        'LoanAmount': loan_amount,
        'CoapplicantIncome': coapplicant_income,
        'Dependents': dependents,
        'Property_Area': property_area,
        'Credit_History': credit_history,
        'Loan_Amount_Term': loan_amount_term
    }

    # Create a DataFrame from the dictionary
    input_df = pd.DataFrame(user_input, index=[0])
    
    st.markdown("<h4 style='text-align: left; color: #D36B00;'>USER INPUT</h4>", unsafe_allow_html=True)
    # Display the input DataFrame
    st.write(input_df)



        # Preprocess the input data
    categoricals = input_df.select_dtypes(include = ['object', 'category'])
    numericals = input_df.select_dtypes(include = 'number')
        
        # Standard Scale the Input Variable.
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    scaler = StandardScaler()
    encoder = LabelEncoder()

    for i in numericals.columns:
        if i in input_df.columns:
            input_df[i] = scaler.fit_transform(input_df[[i]])
    for i in categoricals.columns:
        if i in input_df.columns: 
            input_df[i] = encoder.fit_transform(input_df[i])


    if st.button("Predict Loan Approval"):
        # Make prediction
        prediction = model.predict(input_df)

        # Display prediction
        st.success(f"The loan is {'Approved' if prediction[0] == 1 else 'Rejected'}.")
        st.image('pngwing.com (21).png', width = 100)

# Display content based on the selected page
if selected_page == "Home":
    home_page()
elif selected_page == "Prediction":
    prediction_page()
