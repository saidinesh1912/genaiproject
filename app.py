# app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import lime
import lime.lime_tabular

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(page_title="Linear Regression & LIME", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #ff6600;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #333333;
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #ffcc00;
        color: black;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Linear Regression with LIME</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive app to visualize regression, outliers, and explanations</div>', unsafe_allow_html=True)

# -------------------------
# Generate Synthetic Data
# -------------------------
np.random.seed(42)
X = np.random.uniform(0, 10, size=100).reshape(-1, 1)
noise = np.random.normal(0, 1, size=100)
y = 2 * X.flatten() + noise

# -------------------------
# Train Linear Regression
# -------------------------
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# -------------------------
# Evaluate Model
# -------------------------
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# -------------------------
# Select Outlier
# -------------------------
residuals = np.abs(y - y_pred)
outlier_idx = np.argmax(residuals)

st.write(f"**Default Outlier Point:** X={X[outlier_idx][0]:.2f}, y={y[outlier_idx]:.2f}, predicted={y_pred[outlier_idx]:.2f}")

# Optional: let user select outlier index
idx_slider = st.slider("Select an index to highlight as outlier:", 0, 99, outlier_idx)
selected_outlier = X[idx_slider].reshape(1, -1)

# -------------------------
# Apply LIME
# -------------------------
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X,
    training_labels=y,
    mode="regression",
    feature_names=["X"],
    discretize_continuous=True
)

exp = explainer.explain_instance(
    data_row=selected_outlier.flatten(),
    predict_fn=model.predict
)

# -------------------------
# Plot Results
# -------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X, y, color="blue", label="Data with noise")
ax.plot(X, y_pred, color="red", label="Linear Regression")
ax.scatter(selected_outlier, y[idx_slider], color="green", s=120, label="Selected Outlier", edgecolors='black')
ax.set_xlabel("X")
ax.set_ylabel("y")
ax.set_title("Linear Regression with Outlier Highlighted")
ax.legend()
st.pyplot(fig)

# -------------------------
# Show LIME Explanation
# -------------------------
st.subheader("LIME Explanation for Selected Outlier")
st.write(exp.as_list())
