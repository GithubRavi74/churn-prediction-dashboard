import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ---------- Page Title ----------
st.markdown(
    """
    <h2 style='text-align: center; color: #1F77B4;'>
        📂 THE NEW NTTIS AI Solution - Customer Churn Prediction Dashboard
    </h2>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)


#Streamlit decorator used to cache expensive resources like-Database connections,ML model loading,API clients,Heavy libraries or tools.
#This allows those resources to load once and persist across reruns, making your app faster and more efficient.

@st.cache_resource

def load_model():
    return joblib.load("churn_pipeline.pkl")  # Adjust path if needed
model = load_model()

# ---------- Load Sample Data ----------
@st.cache_data
def load_sample_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    return df.head(5)

sample_df = load_sample_data()


#--------------Preview sample data ------
with st.expander("📄 Sample Data Format (for your CSV upload)"):
        st.markdown(
            "<div style='color:#1F77B4; font-size:18px; font-weight:bold; margin-bottom:10px;'>📊 Sample Format Preview</div>",
            unsafe_allow_html=True
        )
        st.dataframe(sample_df)
  #--------------Preview end ------  



# ---------- Upload Box ----------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.container():
        st.markdown(
            """
            <div style="
                border: 2px solid #4CAF50;
                padding: 25px;
                border-radius: 12px;
                background-color: #E8F5E9;
                text-align: center;
                margin-bottom: 20px;
            ">
                <h5 style='color:green; font-size:20px; font-weight:bold; margin-bottom: 10px;'>
                    SELECT YOUR FILE FOR UPLOAD 👇
                </h5>
            </div>
            """,
            unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader("", type=["csv"])

# ---------- File Upload Handling ----------
user_df = None

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        df_pred = user_df.copy()
     
        if "customerID" in user_df.columns:
            user_df.drop("customerID", axis=1, inplace=True)

        st.success(f"✅ File '{uploaded_file.name}' uploaded successfully!")
        st.markdown("### 🔍 Uploaded File Preview")
        st.dataframe(user_df.head(5))

        # ---------- Run Prediction Button ----------
        if st.button("🚀 Run Prediction & Show Summary"):
            st.markdown("## 📈 Summary Statistics")
            st.dataframe(user_df.describe())

            
            # --- Predict with real model ---
            try:
                
                predictions = model.predict(df_pred)
                user_df["Churn_Prediction"] = predictions
                st.success("✅ Predictions generated!")

                # Download button
                csv = user_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Predictions as CSV",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                    help="Download your predictions file"
                )

            except Exception as e:
                st.error("❌ Prediction failed. Check if columns match model input.")
                st.exception(e)

            # --- Optional Plots ---
            cat_cols = user_df.select_dtypes(include="object").columns.tolist()
            num_cols = user_df.select_dtypes(include=["int", "float"]).columns.tolist()

            if cat_cols:
                st.markdown("### 📊 Categorical Feature Distributions")
                for col in cat_cols:
                    st.markdown(f"**{col}**")
                    st.bar_chart(user_df[col].value_counts())

            if num_cols:
                st.markdown("### 📉 Numerical Feature Distributions")
                for col in num_cols:
                    st.markdown(f"**{col}**")
                    fig, ax = plt.subplots()
                    user_df[col].hist(bins=20, edgecolor="black", ax=ax)
                    ax.set_title(f"Histogram of {col}")
                    st.pyplot(fig)



    except Exception as e:
        st.error("❌ Error reading file. Please upload a valid CSV.")
        st.exception(e)

# ---------- Fallback Sample Format ----------


# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 14px;'>"
    "Developed at NTTIS  | Max file size: 200MB"
    "</div>",
    unsafe_allow_html=True
)
