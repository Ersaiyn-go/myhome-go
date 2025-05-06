import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Заголовок
st.title('AI-прогноз недвижимости в Казахстане')

# Ввод данных пользователем
st.subheader('Введите параметры квартиры:')

city = st.selectbox('Город', ['Almaty', 'Astana', 'Shymkent', 'Karaganda'])
rooms = st.number_input('Количество комнат', min_value=1, max_value=5, value=2)
location = st.selectbox('Расположение', ['Center', 'Outskirts', 'Near Center'])
area = st.number_input('Площадь (в квадратных метрах)', min_value=50, max_value=500, value=100)
floor = st.number_input('Этаж', min_value=1, max_value=25, value=3)
status = st.selectbox('Статус недвижимости', ['Ready to Move', 'Under Construction'])
floor_quality = st.selectbox('Качество этажа', ['Normal', 'Not Very Good', 'Good'])

# Label Encoding
le_location = LabelEncoder().fit(['Center', 'Outskirts', 'Near Center'])
le_status = LabelEncoder().fit(['Ready to Move', 'Under Construction'])
le_city = LabelEncoder().fit(['Almaty', 'Astana', 'Shymkent', 'Karaganda'])
le_floor_quality = LabelEncoder().fit(['Normal', 'Not Very Good', 'Good'])

location_encoded = le_location.transform([location])[0]
status_encoded = le_status.transform([status])[0]
city_encoded = le_city.transform([city])[0]
floor_quality_encoded = le_floor_quality.transform([floor_quality])[0]

input_data = np.array([[rooms, location_encoded, area, floor, floor_quality_encoded, status_encoded, city_encoded]])

# Загрузка и подготовка датасета
df = pd.read_csv('real_estate_database_with_floor.csv')
df['Location'] = le_location.transform(df['Location'])
df['Status'] = le_status.transform(df['Status'])
df['Floor Quality'] = le_floor_quality.transform(df['Floor Quality'])
df['City'] = le_city.transform(df['City'])

X = df[['Rooms', 'Location', 'Area (sqm)', 'Floor', 'Floor Quality', 'Status', 'City']]
y = df['Price (in tenge)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модели Supervised Learning
models = {
    "Linear Regression": LinearRegression(),
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Naive Bayes": GaussianNB(),
    "K-Nearest Neighbor": KNeighborsRegressor(),
    "SVM": SVR(),
    "Gradient Boosting": GradientBoostingRegressor()
}

st.subheader("📌 Алгоритмы машинного обучения (часть A):")

for name, model in models.items():
    if st.button(name):
        try:
            model.fit(X_train, y_train)
            prediction = model.predict(input_data)[0]
            mse = mean_squared_error(y_test, model.predict(X_test))
            st.success(f"Модель: {name}")
            st.write(f"🔹 Предсказанная цена: **{prediction:,.2f} ₸**")
            # st.write(f"📉 Среднеквадратичная ошибка (MSE): {mse:,.2f}")
        except Exception as e:
            st.error(f"Ошибка при запуске модели {name}: {e}")

st.subheader("📊 Сравнение нескольких моделей")

import matplotlib.pyplot as plt

# === Желание пользователя ===
st.subheader("💸 Желаемая цена квартиры (по вашему мнению)")

desired_price = st.number_input(
    "Введите желаемую цену продажи квартиры (в тенге):",
    min_value=1000000,
    max_value=200000000,
    value=30000000,
    step=1000000
)

if st.button("📊 Сравнение всех алгоритмов с желаемой ценой"):
    all_models = {
        "Linear Regression": LinearRegression(),
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Naive Bayes": GaussianNB(),
        "K-Nearest Neighbor": KNeighborsRegressor(),
        "SVM": SVR(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(y=desired_price, color='red', linestyle='--', label="Желаемая цена")

    user_preds = {}

    for i, (name, model) in enumerate(all_models.items()):
        try:
            model.fit(X_train, y_train)
            user_price = model.predict(input_data)[0]
            user_preds[name] = user_price

            # Рисуем звёздочку
            ax.scatter(i, user_price, marker='*', s=250, label=name, edgecolors='black')
            ax.text(i, user_price + 500000, f"{int(user_price):,}", ha='center', fontsize=9)
        except Exception as e:
            st.warning(f"⚠️ {name} — ошибка: {e}")

    ax.set_xticks(range(len(user_preds)))
    ax.set_xticklabels(user_preds.keys(), rotation=45, ha='right')
    ax.set_ylabel("Предсказанная цена (тенге)")
    ax.set_title("📈 Предсказания всех алгоритмов vs Ваша желаемая цена")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    # Кто ближе к желаемой цене
    diffs = {name: abs(desired_price - price) for name, price in user_preds.items()}
    closest_model = min(diffs, key=diffs.get)
    closest_diff = diffs[closest_model]

    st.markdown("---")
    st.markdown(f"🎯 **Ближе всего к вашей желаемой цене**: `{closest_model}` "
                f"(отклонение: **{closest_diff:,.0f} ₸**)")


# ========== PART B ==========
st.subheader("🧠 Алгоритмы без учителя (часть B):")

# K-Means
if st.button("K-Means Clustering"):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    cluster = kmeans.predict(input_data)[0]
    st.success(f"K-Means определил: квартира относится к кластеру №{cluster + 1}")

# PCA
if st.button("PCA (Снижение размерности)"):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    user_transformed = pca.transform(input_data)
    st.success("PCA-трансформация завершена:")
    st.write(f"📉 Компонента 1: {user_transformed[0][0]:.2f}")
    st.write(f"📉 Компонента 2: {user_transformed[0][1]:.2f}")

# Маппинг значений обратно в читаемый вид
city_map = dict(zip(le_city.transform(le_city.classes_), le_city.classes_))
status_map = dict(zip(le_status.transform(le_status.classes_), le_status.classes_))
rooms_map = {str(i): f"{i} комнат" for i in range(1, 6)}

combined_map = {
    **{str(k): f"Город = {v}" for k, v in city_map.items()},
    **{str(k): f"Статус = {v}" for k, v in status_map.items()},
    **rooms_map
}

# Apriori кнопка
if st.button("Apriori (Ассоциации)"):
    df_apriori = df[['City', 'Rooms', 'Status']].astype(str)
    transactions = df_apriori.values.tolist()

    te = TransactionEncoder()
    te_data = te.fit_transform(transactions)
    df_trans = pd.DataFrame(te_data, columns=te.columns_)

    freq_items = apriori(df_trans, min_support=0.05, use_colnames=True)
    rules = association_rules(freq_items, metric="lift", min_threshold=1)

    st.subheader("📊 Частые ассоциации:")

    if not rules.empty:
        for i, row in rules.head(3).iterrows():
            antecedents = [combined_map.get(item, item) for item in row['antecedents']]
            consequents = [combined_map.get(item, item) for item in row['consequents']]
            st.write(f"✅ Если {' и '.join(antecedents)}, то обычно {' и '.join(consequents)} "
                     f"(правило встречается в {row['support']*100:.0f}% записей)")
    else:
        st.warning("Правила с заданными порогами не найдены.")


if st.button("📈 Показать график кластеров (PCA + KMeans)"):
    # PCA
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    centers_2d = pca.transform(centers)

    # Преобразуем ввод пользователя
    user_pca = pca.transform(input_data)

    # График
    fig, ax = plt.subplots()
    scatter = ax.scatter(components[:, 0], components[:, 1], c=labels, cmap='viridis', alpha=0.6, label='Квартиры')
    ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X', s=200, label='Центры кластеров')
    ax.scatter(user_pca[0][0], user_pca[0][1], c='black', marker='*', s=200, label='Ваша квартира')

    ax.set_title('PCA + K-Means Кластеры')
    ax.set_xlabel('Компонента 1')
    ax.set_ylabel('Компонента 2')
    ax.legend()
    st.pyplot(fig)