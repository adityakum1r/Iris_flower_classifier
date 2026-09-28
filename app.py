import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("Iris.csv")


# -----------------------------
# Train Model
# -----------------------------

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df["Species"])

X = df.drop(columns=["Id", "Species"])


model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X, y)


# -----------------------------
# Application UI
# -----------------------------

st.title("🌸 Iris Flower Classifier")

st.write(
    "Enter the measurements of an Iris flower "
    "to predict its species."
)

st.divider()


# -----------------------------
# User Inputs
# -----------------------------

st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)


with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )


with col2:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )


st.divider()


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔮 Predict Species"):

    input_data = pd.DataFrame({
        "SepalLengthCm": [sepal_length],
        "SepalWidthCm": [sepal_width],
        "PetalLengthCm": [petal_length],
        "PetalWidthCm": [petal_width]
    })

    prediction = model.predict(input_data)

    predicted_species = label_encoder.inverse_transform(
        prediction
    )[0]

    st.success(
        f"🌸 Predicted Species: {predicted_species}"
    )