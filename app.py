import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

model = pk.load(open('Prediction_Model.pkl', 'rb'))

st.header('Car Price Prediction System')

df = pd.read_csv('Car_Details.csv')

def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

df['name'] = df['name'].apply(get_brand_name)

name = st.selectbox('Select Car Brand', df['name'].unique())
year = st.slider('Car Manufactured Year', 1994, 2024)
km_driven = st.slider('No of kms Driven', 11, 200000)
fuel = st.selectbox('Fuel type', df['fuel'].unique())
seller_type = st.selectbox('Seller type', df['seller_type'].unique())
transmission = st.selectbox('Transmission type', df['transmission'].unique())
owner = st.selectbox('Owner type', df['owner'].unique())

if st.button("Predict"):
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
    )

    input_data_model['owner'] = input_data_model['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5]
    )

    input_data_model['fuel'] = input_data_model['fuel'].replace(
        ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'],
        [1, 2, 3, 4, 5]
    )

    input_data_model['seller_type'] = input_data_model['seller_type'].replace(
        ['Individual', 'Dealer', 'Trustmark Dealer'],
        [1, 2, 3]
    )

    input_data_model['transmission'] = input_data_model['transmission'].replace(
        ['Manual', 'Automatic'],
        [1, 2]
    )

    input_data_model['name'] = input_data_model['name'].replace(
        ['Maruti', 'Hyundai', 'Datsun', 'Honda', 'Tata', 'Chevrolet', 'Toyota', 'Jaguar',
         'Mercedes-Benz', 'Audi', 'Skoda', 'Jeep', 'BMW', 'Mahindra', 'Ford', 'Nissan',
         'Renault', 'Fiat', 'Volkswagen', 'Volvo', 'Mitsubishi', 'Land', 'Daewoo', 'MG',
         'Force', 'Isuzu', 'OpelCorsa', 'Ambassador', 'Kia'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    )

    car_price = model.predict(input_data_model)

    st.markdown('Car Price is going to be ' + str(car_price[0]))