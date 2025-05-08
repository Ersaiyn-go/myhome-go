import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torchvision import models, transforms
from ersalib import elinearRegression

# ======== Real Estate Model ========
df = pd.read_csv('real_estate_database_with_floor.csv')

label_encoder = LabelEncoder()
df['Location'] = label_encoder.fit_transform(df['Location'])
df['Status'] = label_encoder.fit_transform(df['Status'])
df['Floor Quality'] = label_encoder.fit_transform(df['Floor Quality'])
df['City'] = label_encoder.fit_transform(df['City'])

X = df[['Rooms', 'Location', 'Area (sqm)', 'Floor Quality', 'Status', 'City']]
y = df['Price (in tenge)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = elinearRegression()
model.fit(X_train.values.tolist(), y_train.values.tolist())

# ======== Window View Classifier ========
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
st.title('🏠 Real Estate Price Estimation with Window View')

city = st.selectbox('City', ['Almaty', 'Astana', 'Shymkent', 'Karaganda'])
rooms = st.number_input('Number of rooms', min_value=1, max_value=5, value=2)
location = st.selectbox('Location', ['Center', 'Outskirts', 'Near Center'])
area = st.number_input('Area (sqm)', min_value=50, max_value=5000, value=100)
status = st.selectbox('Property status', ['Under Construction', 'Ready to Move'])

uploaded_image = st.file_uploader("📷 Upload window view photo", type=['jpg', 'png', 'jpeg'])

# Encoding
location_encoded = label_encoder.fit(['Center', 'Outskirts', 'Near Center']).transform([location])[0]
status_encoded = label_encoder.fit(['Under Construction', 'Ready to Move']).transform([status])[0]
city_encoded = label_encoder.fit(['Almaty', 'Astana', 'Shymkent', 'Karaganda']).transform([city])[0]
floor_quality_encoded = label_encoder.fit(['Normal', 'Not Very Good', 'Good']).transform(['Normal'])[0]

input_data = np.array([[rooms, location_encoded, area, floor_quality_encoded, status_encoded, city_encoded]])

if st.button('Estimate Price'):

    predicted_price = model.predict(input_data.tolist())[0]

    if uploaded_image:
        img = Image.open(uploaded_image).convert('RGB')
        view_type = predict_view(img)

        # View-based price adjustment
        adjustments = {
            'mountain': 1.15,
            'park': 1.10,
            'water': 1.12,
            'city': 1.00,
            'yard': 0.95,
            'construction': 0.90
        }
        adjusted_price = predicted_price * adjustments.get(view_type, 1.0)

        st.image(img, caption=f'Detected view: {view_type}', use_column_width=True)
        st.write(f'🧠 Detected window view: **{view_type}**')
        st.write(f'💰 Final price with view adjustment: **{adjusted_price:,.2f} ₸**')
    else:
        st.warning("Please upload a window view photo for accurate estimation.")
        st.write(f'Base predicted price: **{predicted_price:,.2f} ₸**')