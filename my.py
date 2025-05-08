import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 👉 Import custom algorithms
from ersalib import elinearRegression, elogisticRegression, edecisionTreeClassifier, erandomForestClassifier, enaiveBayes, ekNNClassifier, eSVM, eGradientBoosting

st.title('🏗️ AI Real Estate Price Predictor (Custom Models)')

# User input
city = st.selectbox('City', ['Almaty', 'Astana', 'Shymkent', 'Karaganda'])
rooms = st.number_input('Number of rooms', 1, 5, value=2)
location = st.selectbox('Location', ['Center', 'Outskirts', 'Near Center'])
area = st.number_input('Area (sqm)', 50, 500, value=100)
floor = st.number_input('Floor', 1, 25, value=3)
status = st.selectbox('Property status', ['Ready to Move', 'Under Construction'])
floor_quality = st.selectbox('Floor quality', ['Normal', 'Not Very Good', 'Good'])

# Encoding
le_location = LabelEncoder().fit(['Center', 'Outskirts', 'Near Center'])
le_status = LabelEncoder().fit(['Ready to Move', 'Under Construction'])
le_city = LabelEncoder().fit(['Almaty', 'Astana', 'Shymkent', 'Karaganda'])
le_floor_quality = LabelEncoder().fit(['Normal', 'Not Very Good', 'Good'])

location_encoded = le_location.transform([location])[0]
status_encoded = le_status.transform([status])[0]
city_encoded = le_city.transform([city])[0]
floor_quality_encoded = le_floor_quality.transform([floor_quality])[0]

input_data = [[rooms, location_encoded, area, floor, floor_quality_encoded, status_encoded, city_encoded]]

# Load data
st.subheader("📁 Loading and preparing dataset")
df = pd.read_csv("real_estate_database_with_floor.csv")
df['Location'] = le_location.transform(df['Location'])
df['Status'] = le_status.transform(df['Status'])
df['Floor Quality'] = le_floor_quality.transform(df['Floor Quality'])
df['City'] = le_city.transform(df['City'])

X = df[['Rooms', 'Location', 'Area (sqm)', 'Floor', 'Floor Quality', 'Status', 'City']]
y = df['Price (in tenge)']
X_train, X_test, y_train, y_test = train_test_split(X.values.tolist(), y.values.tolist(), test_size=0.2, random_state=42)

# Custom models
st.subheader("📌 Custom Regression Algorithms")
models = {
    "My Linear Regression": elinearRegression(),
    "My Decision Tree": edecisionTreeClassifier(),
    "My Random Forest": erandomForestClassifier(),
    "My Naive Bayes": enaiveBayes(),
    "My KNN": ekNNClassifier(),
    "My Gradient Boosting": eGradientBoosting()
}

for name, model in models.items():
    if st.button(name):
        try:
            model.fit(X_train, y_train)
            prediction = model.predict(input_data)[0]
            st.success(f"✅ {name}")
            st.write(f"💰 Predicted price: **{prediction:,.0f} ₸**")
        except Exception as e:
            st.error(f"Error in {name}: {e}")

# Desired price
st.subheader("💸 Desired Apartment Price")
desired_price = st.number_input("Enter your desired price:", 1_000_000, 200_000_000, 30_000_000, step=1_000_000)

if st.button("📊 Compare all models"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(y=desired_price, color='red', linestyle='--', label="Desired Price")

    results = {}
    for i, (name, model) in enumerate(models.items()):
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(input_data)[0]
            results[name] = y_pred
            ax.scatter(i, y_pred, s=200, marker='*', label=name)
            ax.text(i, y_pred + 500_000, f"{int(y_pred):,}", ha='center', fontsize=9)
        except Exception as e:
            st.warning(f"{name} — error: {e}")

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(results.keys(), rotation=45, ha='right')
    ax.set_ylabel("Predicted Price (₸)")
    ax.set_title("Custom Model Predictions vs Your Desired Price")
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    diffs = {name: abs(desired_price - p) for name, p in results.items()}
    best_model = min(diffs, key=diffs.get)
    st.markdown(f"🎯 Closest prediction by `{best_model}` (diff: {diffs[best_model]:,.0f} ₸)")


# ========== PART B ==========
st.subheader("🧠 Unsupervised Learning (Part B, Custom)")

from ersalib import eKMeans, ePCA, eApriori

# K-Means (custom)
if st.button("K-Means Clustering (custom)"):
    kmeans = eKMeans(k=3)
    kmeans.fit(X.values.tolist())
    cluster = kmeans.predict(input_data)[0]
    st.success(f"Custom K-Means: this apartment belongs to cluster #{cluster + 1}")

# PCA (custom)
if st.button("PCA (Dimensionality Reduction, custom)"):
    pca = ePCA(n_components=2)
    pca.fit(X.values.tolist())
    transformed = pca.transform(X.values.tolist())
    user_point = pca.transform(input_data)[0]

    st.success("PCA transformation completed:")
    st.write(f"📉 Component 1: {user_point[0]:.2f}")
    st.write(f"📉 Component 2: {user_point[1]:.2f}")

    fig, ax = plt.subplots()
    ax.scatter([p[0] for p in transformed], [p[1] for p in transformed], alpha=0.5, label='Apartments')
    ax.scatter(user_point[0], user_point[1], color='red', marker='*', s=200, label='Your apartment')
    ax.set_title("PCA Plot: 2D Projection of Apartments")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend()
    st.pyplot(fig)

# Mapping values for display
city_map = dict(zip(le_city.transform(le_city.classes_), le_city.classes_))
status_map = dict(zip(le_status.transform(le_status.classes_), le_status.classes_))
rooms_map = {str(i): f"{i} room(s)" for i in range(1, 6)}

combined_map = {
    **{str(k): f"City = {v}" for k, v in city_map.items()},
    **{str(k): f"Status = {v}" for k, v in status_map.items()},
    **rooms_map
}

# Apriori (custom)
if st.button("Apriori (Association Rules, custom)"):
    df_apriori = df[['City', 'Rooms', 'Status']].astype(str)
    transactions = df_apriori.values.tolist()

    model = eApriori(min_support=0.05)
    model.fit(transactions)
    freq_items = model.get_freq_itemsets()

    st.subheader("📊 Frequent Association Rules:")

    count = 0
    for itemset, support in freq_items:
        if len(itemset) > 1:
            antecedent = list(itemset)[:-1]
            consequent = list(itemset)[-1:]
            antecedent_str = [combined_map.get(a, a) for a in antecedent]
            consequent_str = [combined_map.get(c, c) for c in consequent]
            st.write(f"✅ If {' and '.join(antecedent_str)}, then usually {' and '.join(consequent_str)} "
                     f"(appears in {support * 100:.0f}% of records)")
            count += 1
        if count == 3:
            break

    if count == 0:
        st.warning("No rules found for the given thresholds.")


# ========== PART C: Binary Price Classification ========== 
st.subheader("🧪 Classification: Expensive or Not")

from ersalib import elogisticRegression, eSVM, elinearRegression

# Threshold for expensive apartment classification
threshold_price = st.slider("Threshold for 'expensive' apartment (₸):", 10_000_000, 100_000_000, 30_000_000, step=1_000_000)

# Classification using elogisticRegression
if st.button("Logistic Regression: Expensive or Not"):
    y_class = (y >= threshold_price).astype(int)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X.values.tolist(), y_class.tolist(), test_size=0.2, random_state=42)
    model = elogisticRegression()
    model.fit(X_train_cls, y_train_cls)
    result = model.predict(input_data)[0]
    st.success("Model: Logistic Regression")
    st.write("🔍 Prediction:", "💰 Expensive apartment" if result == 1 else "💡 Not expensive")

# Classification using eSVM
if st.button("SVM: Expensive or Not"):
    y_class = (y >= threshold_price).astype(int)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X.values.tolist(), y_class.tolist(), test_size=0.2, random_state=42)
    model = eSVM()
    model.fit(X_train_cls, y_train_cls)
    result = model.predict(input_data)[0]
    st.success("Model: SVM")
    st.write("🔍 Prediction:", "💰 Expensive apartment" if result == 1 else "💡 Not expensive")

