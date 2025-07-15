import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from xgboost import XGBClassifier

# ---------- Page Title ----------
st.markdown(
    """
    <h2 style='text-align: center; color: #1F77B4;'>
        📂 ALL NEW NTTIS AI Solution - Customer Churn Prediction Dashboard
    </h2>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        with open("churn_pipeline.pkl", "rb") as f:
            return cloudpickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model: {type(e).__name__}: {e}")
        raise

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
        #if st.button("🚀 Run Prediction & Show Summary"):

        if "run_pred" not in st.session_state:
            st.session_state.run_pred = False

        if st.button("🚀 Run Prediction & Show Summary"):
            st.session_state.run_pred = True

        if st.session_state.run_pred:
    
            st.markdown("## 📈 Overview of uploaded data- A Numerical Insight")
            view_option = st.radio(
                "Choose how to view numerical insights:",
                ('Summary Table', 'Box Plots')
            )

            num_cols = user_df.select_dtypes(include=["int", "float"]).columns.tolist()

            if view_option == 'Summary Table':
                st.dataframe(user_df[num_cols].describe().T)
            else:
                st.write("### Box Plots for Numerical Features")
                for col in num_cols:
                    fig, ax = plt.subplots()
                    user_df.boxplot(column=col, ax=ax)
                    ax.set_title(f'Boxplot of {col}')
                    st.pyplot(fig)

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
                st.markdown("### 📊 Categorical Feature DistributionsXXX")
                #COMMENTING TEMPORARILY THE STREAMLIT VISUALIZATIONS FOR CATEGORICAL FEATURE DISTRIBUTION
                #for col in cat_cols:
                    #st.markdown(f"**{col}**")
                    #st.bar_chart(user_df[col].value_counts())
                
                #MATPLOTLIB VISUALIZATION
                import matplotlib.pyplot as plt
                for col in cat_cols:
                    st.markdown(f"**{col}**")
                    fig, ax = plt.subplots(figsize=(4, 3))  # Control figure size
                    value_counts = user_df[col].value_counts()
                    ax.bar(value_counts.index, value_counts.values, width=0.4)  # Set bar width < 1
                    ax.set_xlabel(col)
                    ax.set_ylabel("Count")
                    ax.set_title(f"Distribution of {col}")
                    st.pyplot(fig)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                #SEABORN VISUALIZATION
                import seaborn as sns
                for col in cat_cols:
                    st.markdown(f"**{col}**")
                    fig, ax = plt.subplots(figsize=(4, 3))
                    sns.countplot(x=col, data=user_df, ax=ax, width=0.4)  # width < 1 for thinner bars
                    ax.set_title(f"Distribution of {col}")
                    st.pyplot(fig)




            if num_cols:
                st.markdown("### 📉 Numerical Feature Distributions")
                for col in num_cols:
                    st.markdown(f"**{col}**")
                    fig, ax = plt.subplots()
                    user_df[col].hist(bins=20, edgecolor="black", ax=ax)
                    ax.set_title(f"Histogram of {col}")
                    st.pyplot(fig)

            # ---------- Churn Distribution ----------
            if "Churn_Prediction" in user_df.columns:
                st.markdown("## 📌 Churn Prediction Summary")
                churn_counts = user_df["Churn_Prediction"].value_counts().rename(index={0: "No Churn", 1: "Churn"})
                st.bar_chart(churn_counts)

                # MonthlyCharges vs Churn
                if "MonthlyCharges" in user_df.columns:
                    st.markdown("### 📊 MonthlyCharges by Churn")
                    fig1, ax1 = plt.subplots()
                    for label, group in user_df.groupby("Churn_Prediction"):
                        ax1.hist(group["MonthlyCharges"], bins=30, alpha=0.5, label=f"{'Churn' if label==1 else 'No Churn'}", edgecolor='black')
                    ax1.set_xlabel("MonthlyCharges")
                    ax1.set_ylabel("Count")
                    ax1.set_title("MonthlyCharges Distribution by Churn")
                    ax1.legend()
                    st.pyplot(fig1)

                # Tenure vs Churn
                if "tenure" in user_df.columns:
                    st.markdown("### 📊 Tenure by Churn")
                    fig2, ax2 = plt.subplots()
                    for label, group in user_df.groupby("Churn_Prediction"):
                        ax2.hist(group["tenure"], bins=30, alpha=0.5, label=f"{'Churn' if label==1 else 'No Churn'}", edgecolor='black')
                    ax2.set_xlabel("Tenure (Months)")
                    ax2.set_ylabel("Count")
                    ax2.set_title("Tenure Distribution by Churn")
                    ax2.legend()
                    st.pyplot(fig2)

    except Exception as e:
        st.error("❌ Error reading file. Please upload a valid CSV.")
        st.exception(e)

# ---------- Footer ----------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: grey; font-size: 14px;'>"
    "Developed at NTTIS  | Max file size: 200MB"
    "</div>",
    unsafe_allow_html=True
)
