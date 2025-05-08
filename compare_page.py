import pandas as pd
import numpy as np
import time
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from ersalib import elinearRegression

st.title("📊 Сравнение линейной регрессии: своя vs sklearn")

# Загрузка данных
st.subheader("📁 Загрузка и подготовка данных")
df = pd.read_csv("real_estate_database_with_floor.csv")

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Status'] = label_encoder.fit_transform(df['Status'])
df['Floor Quality'] = label_encoder.fit_transform(df['Floor Quality'])
df['City'] = label_encoder.fit_transform(df['City'])

X = df[['Rooms', 'Location', 'Area (sqm)', 'Floor Quality', 'Status', 'City']]
y = df['Price (in tenge)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.subheader("🏡 Введите параметры квартиры")
city = st.selectbox('Город', ['Almaty', 'Astana', 'Shymkent', 'Karaganda'])
rooms = st.number_input('Количество комнат', min_value=1, max_value=5, value=2)
location = st.selectbox('Расположение', ['Center', 'Outskirts', 'Near Center'])
area = st.number_input('Площадь (в квадратных метрах)', min_value=50, max_value=5000, value=100)
status = st.selectbox('Статус недвижимости', ['Under Construction', 'Ready to Move'])

location_encoded = label_encoder.fit(['Center', 'Outskirts', 'Near Center']).transform([location])[0]
status_encoded = label_encoder.fit(['Under Construction', 'Ready to Move']).transform([status])[0]
city_encoded = label_encoder.fit(['Almaty', 'Astana', 'Shymkent', 'Karaganda']).transform([city])[0]
floor_quality_encoded = label_encoder.fit(['Normal', 'Not Very Good', 'Good']).transform(['Normal'])[0]

input_data = [[rooms, location_encoded, area, floor_quality_encoded, status_encoded, city_encoded]]

if st.button("🔍 Сравнить модели"):
    results = []

    # Sklearn
    start = time.time()
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    sk_preds = sk_model.predict(X_test)
    sk_price = sk_model.predict(input_data)[0]
    pred_time = time.time() - start

    sk_score = r2_score(y_test, sk_preds)
    results.append(["Sklearn LinearRegression", sk_score, train_time, pred_time, sk_price])

    # Своя
    start = time.time()
    my_model = elinearRegression()
    my_model.fit(X_train.values.tolist(), y_train.values.tolist())
    train_time = time.time() - start

    start = time.time()
    my_preds = my_model.predict(X_test.values.tolist())
    my_price = my_model.predict(input_data)[0]
    pred_time = time.time() - start

    my_score = r2_score(y_test, my_preds)
    results.append(["Моя elinearRegression", my_score, train_time, pred_time, my_price])

    # Выводим
    st.subheader("📈 Результаты сравнения")
    df_results = pd.DataFrame(results, columns=["Модель", "R²", "Время обучения (сек)", "Время предсказания (сек)", "Предсказанная цена (тенге)"])
    st.dataframe(df_results)

    st.bar_chart(df_results.set_index("Модель")["R²"])