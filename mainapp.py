import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder
import gdown as gd
import os

# Use st.cache_resource to cache the model loading
@st.cache_resource
def load_model():
    return load('model.joblib')

model = load_model()

# Data Preprocessing Code

# Define the path where the file will be downloaded
file_path = 'all_car_adverts.csv'

# Check if the file already exists, if not, download it
if not os.path.exists(file_path):
    # Download dataset (only the first time)
    url = "https://drive.google.com/uc?id=1VWcklFoYkOr2ZYqnZREBWum-2KtyM75Y&export=download"
    gd.download(url, file_path, quiet=False)

# Now load the CSV file
newfile = pd.read_csv(file_path)

# Continue with the preprocessing steps as in your original code
drop_cols = ['Unnamed: 0', 'make', 'variant', 'car_badges', 'car_sub_title', 'car_attention_grabber',
             'car_specs', 'car_seller_rating', 'reg', 'body_type', 'engine_size_unit', 'ulez',
             'car_seller_location', 'full_service', 'part_service', 'part_warranty', 'full_dealership',
             'first_year_road_tax', 'brand_new', 'finance_available', 'discounted']
newfile.drop(columns=drop_cols, axis=1, errors='ignore', inplace=True)

title_counts = newfile.groupby('car_title')['model'].nunique()
titles_to_remove = title_counts[title_counts > 1].index
newfile = newfile[~newfile['car_title'].isin(titles_to_remove)]

for col in ['car_title', 'model', 'car_seller', 'transmission', 'fuel_type']:
    newfile[col] = newfile[col].fillna(newfile[col].mode()[0])
for col in ['miles', 'engine_vol', 'engine_size', 'num_owner']:
    newfile[col] = newfile[col].fillna(newfile[col].mean())
newfile['year'] = pd.to_numeric(newfile['year'], errors='coerce')
newfile['year'] = newfile['year'].fillna(newfile['year'].mean())

# Encode categorical variables
label_encoders = {}
original_labels = {}
high_cardinality_cols = ['car_title', 'model', 'car_seller']
for col in high_cardinality_cols:
    le = LabelEncoder()
    newfile[col] = le.fit_transform(newfile[col])
    label_encoders[col] = le
    original_labels[col] = dict(zip(le.transform(le.classes_), le.classes_))

low_cardinality_cols = ['transmission', 'fuel_type']
newfile = pd.get_dummies(newfile, columns=low_cardinality_cols, drop_first=True)

# Streamlit App
st.title('Car Price Prediction')

# Car Title Selection
car_titles = list(original_labels['car_title'].values())
car_title_selected = st.selectbox("Select Car", car_titles)

# Dynamically filter models based on selected car title
selected_car_index = list(original_labels['car_title'].values()).index(car_title_selected)
selected_car_label = list(original_labels['car_title'].keys())[selected_car_index]
filtered_models = newfile[newfile['car_title'] == selected_car_label]['model'].unique()

# Model Selection
model_options = [original_labels['model'][m] for m in filtered_models]
model_selected = st.selectbox("Model", model_options)

# Other Inputs
car_seller_selected = st.selectbox("Car Seller", list(original_labels['car_seller'].values()))
transmission = st.selectbox('Transmission', ['Automatic', 'Manual'])
fuel_type = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Electric'])
miles = st.number_input('Mileage (in miles)', min_value=0, step=1)
year = st.number_input('Year of Manufacture', min_value=2000, max_value=2025)
engine_vol = st.number_input('Engine Volume', min_value=0.0, step=0.1)
engine_size = st.number_input('Engine Size', min_value=0.0, step=0.1)
num_owner = st.number_input('Number of Owners', min_value=0, step=1)

def preprocess_input(input_data):
    input_data['car_title'] = label_encoders['car_title'].transform([input_data['car_title']])[0]
    input_data['model'] = label_encoders['model'].transform([input_data['model']])[0]
    input_data['car_seller'] = label_encoders['car_seller'].transform([input_data['car_seller']])[0]
    
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df, columns=low_cardinality_cols, drop_first=True)
    
    missing_cols = set(newfile.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0  
    
    extra_cols = set(input_df.columns) - set(newfile.columns)
    input_df.drop(columns=extra_cols, inplace=True, errors='ignore')
    
    input_df = input_df[newfile.drop(columns=['car_price']).columns]  # Ensure column order matches training data
    
    return input_df

if st.button('Predict Car Price'):
    input_data = {
        'car_title': car_title_selected,
        'model': model_selected,
        'car_seller': car_seller_selected,
        'transmission': transmission,
        'fuel_type': fuel_type,
        'miles': miles,
        'year': year,
        'engine_vol': engine_vol,
        'engine_size': engine_size,
        'num_owner': num_owner
    }
    processed_input = preprocess_input(input_data)
    predicted_price = model.predict(processed_input)
    st.write(f'### The predicted car price is: **£{predicted_price[0]:,.2f}**')
st.markdown("<hr>", unsafe_allow_html=True)
col1, col2 = st.columns([5, 1])  
with col1:
    st.markdown("### Created by Adwaith Kalathuru")
with col2:
    st.image("image0.jpeg", width=100)

