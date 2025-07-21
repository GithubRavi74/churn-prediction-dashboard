# 📦 This is your enhanced Streamlit app with Churn Retention Agent integration

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
import seaborn as sns

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    with open("churn_pipeline.pkl", "rb") as f:
        return cloudpickle.load(f)

model = load_model()

# ---------- Sample Data Preview ----------
@st.cache_data
def load_sample_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    return df.head(5)

sample_df = load_sample_data()

st.title("📂 NTTIS AI: Customer Churn Prediction & Retention Agent")
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📄 Sample CSV Format"):
    st.dataframe(sample_df)

# ---------- Upload File ----------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        user_df = pd.read_csv(uploaded_file)
        df_pred = user_df.copy()
        if "customerID" in user_df.columns:
            user_df.drop("customerID", axis=1, inplace=True)

        st.success("✅ File uploaded successfully!")
        st.dataframe(user_df.head())

        if st.button("🚀 Run Prediction & Generate Retention Plan"):
            # --- Predict ---
            predictions = model.predict(df_pred)
            user_df["Churn_Prediction"] = predictions

            st.success("✅ Predictions complete!")
            st.markdown("### 🔮 Prediction Results")
            st.dataframe(user_df.head())

            # ---------- Retention Message Function ----------
            def generate_retention_message(row):
                contract = row['Contract']
                monthly = row['MonthlyCharges']
                senior = row['SeniorCitizen']
                tenure = row['tenure']
                tech = row['TechSupport']
                name = row.name

                msg = f"Hello Customer {name}, "
                if senior:
                    msg += "as a respected senior member, "

                if tenure > 24:
                    msg += "thank you for being with us for over 2 years! Here's a loyalty upgrade offer. "
                elif tenure <= 3:
                    msg += "we know you're new — let us offer a welcome bonus. "
                else:
                    msg += "we value your time with us. "

                if contract == "Month-to-month":
                    msg += "How about a ₹20 discount on your bill or a 3-month free upgrade? "
                if tech == "No":
                    msg += "We can also add free Tech Support for 3 months. "

                msg += "We're here to keep you happy!"
                return msg

            # ---------- Apply Retention Strategy ----------
            churn_customers = user_df[user_df["Churn_Prediction"] == 1].copy()
            if not churn_customers.empty:
                churn_customers["Retention_Message"] = churn_customers.apply(generate_retention_message, axis=1)

                st.markdown("### 💬 Retention Messages for Predicted Churn Customers")
                st.dataframe(churn_customers[["Churn_Prediction", "MonthlyCharges", "Contract", "TechSupport", "tenure", "Retention_Message"]])

                # Save all predictions + retention to CSV
                final_df = user_df.copy()
                final_df["Retention_Message"] = user_df.apply(
                    lambda row: generate_retention_message(row) if row["Churn_Prediction"] == 1 else "", axis=1
                )

                csv = final_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Full Predictions with Retention Messages",
                    data=csv,
                    file_name="churn_with_retention.csv",
                    mime="text/csv"
                )
            else:
                st.info("🎉 No churn predicted — all customers are likely to stay!")

    except Exception as e:
        st.error("❌ Failed to process file")
        st.exception(e)
