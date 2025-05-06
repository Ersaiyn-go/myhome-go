import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torchvision import models, transforms

# ======== Модель недвижимости ========
df = pd.read_csv('real_estate_database_with_floor.csv')

label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Status'] = label_encoder.fit_transform(df['Status'])
df['Floor Quality'] = label_encoder.fit_transform(df['Floor Quality'])  
df['City'] = label_encoder.fit_transform(df['City'])  

X = df[['Rooms', 'Location', 'Area (sqm)', 'Floor Quality', 'Status', 'City']] 
y = df['Price (in tenge)']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ======== Классификатор вида из окна ========
class_names = ['city', 'construction', 'mountain', 'park', 'water', 'yard']

def predict_view(img: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(img).unsqueeze(0)

    resnet = models.resnet18()
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(class_names))
    resnet.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    resnet.eval()

    with torch.no_grad():
        output = resnet(image)
        _, predicted = torch.max(output, 1)
    
    return class_names[predicted.item()]

# ======== Streamlit UI ========
st.title('🏠 Прогнозирование цены недвижимости с видом из окна')

city = st.selectbox('Город', ['Almaty', 'Astana', 'Shymkent', 'Karaganda'])
rooms = st.number_input('Количество комнат', min_value=1, max_value=5, value=2)
location = st.selectbox('Расположение', ['Center', 'Outskirts', 'Near Center'])
area = st.number_input('Площадь (в квадратных метрах)', min_value=50, max_value=5000, value=100)
status = st.selectbox('Статус недвижимости', ['Under Construction', 'Ready to Move'])

uploaded_image = st.file_uploader("📷 Загрузите фото из окна", type=['jpg', 'png', 'jpeg'])

# Кодирование
location_encoded = label_encoder.fit(['Center', 'Outskirts', 'Near Center']).transform([location])[0]
status_encoded = label_encoder.fit(['Under Construction', 'Ready to Move']).transform([status])[0]
city_encoded = label_encoder.fit(['Almaty', 'Astana', 'Shymkent', 'Karaganda']).transform([city])[0]
floor_quality_encoded = label_encoder.fit(['Normal', 'Not Very Good', 'Good']).transform(['Normal'])[0]

input_data = np.array([[rooms, location_encoded, area, floor_quality_encoded, status_encoded, city_encoded]])

if st.button('Узнать цену'):

    predicted_price = model.predict(input_data)[0]

    if uploaded_image:
        img = Image.open(uploaded_image).convert('RGB')
        view_type = predict_view(img)

        # Коррекция по виду
        adjustments = {
            'mountain': 1.15,
            'park': 1.10,
            'water': 1.12,
            'city': 1.00,
            'yard': 0.95,
            'construction': 0.90
        }
        adjusted_price = predicted_price * adjustments.get(view_type, 1.0)

        st.image(img, caption=f'Распознанный вид: {view_type}', use_column_width=True)
        st.write(f'🧠 Распознанный вид из окна: **{view_type}**')
        st.write(f'💰 Итоговая цена с учетом вида: **{adjusted_price:,.2f} тенге**')
    else:
        st.warning("Пожалуйста, загрузите изображение окна для точной оценки.")
        st.write(f'Предсказанная базовая цена: **{predicted_price:,.2f} тенге**')
