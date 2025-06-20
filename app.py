
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap

model = joblib.load("churn-model.pkl")

st.set_page_config(page_title="NTTIS Customer Churn Prediction AI solution", layout="wide")
st.title("📊 NTTIS Customer Churn Prediction AI solution")

...
st.markdown(
    "<h5 style='color:blue; font-size:18px; font-weight:bold;'>Please upload a data file in CSV format to begin</h5>",
    unsafe_allow_html=True
)
...
#st.markdown("Please upload a data file in CSV format to begin")
 


# Load and display a few rows of the sample dataset
@st.cache_data

def load_sample_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)
    return df.head(5)  # Show only top 5 rows

sample_df = load_sample_data()

#######################################################
#USE THIS CODE IF YOU WANT SAMPLE DATA TO BE SHOWN IN SIDEBAR.
#st.sidebar.header("📄 Sample Data format that you need to follow when uploading your csv")
#st.sidebar.write("Sample input format:")
#st.sidebar.dataframe(sample_df)
#st.sidebar.subheader("Upload Your Data similar to the above format")
#####################################################


######################################################################
# USE THIS CODE IF THE SAMPLE DATA NEEDS TO BE SHOWN AS PULL DOWN MENU
with st.expander("📄 Click here to see the sample Data format that you need to follow when uploading your csv"):
    st.dataframe(sample_df)
    st.write("Upload Your Data similar to the above format")
###########################################################

######################################################################
# USE THIS CODE IF THE SAMPLE DATA NEEDS TO BE SHOWN ALWAYS ON SCREEN
# st.header("📄 Sample Data format that you need to follow when uploading your csv")
# st.dataframe(sample_df)
# st.write("Upload Your Data similar to the above format")

# Streamlit automatically gives file_uploader the full width of the container (default style).
# The "Drag and drop" text appears on the left; the Browse button on the right — taking the entire row width.
# if you want the button to appear in center use this code

st.markdown(
    "<h5 style='color:green; font-size:18px; font-weight:bold;'> SELECT FILE TO UPLOAD 👇 (CSV, Max 200MB)</h5>",
    unsafe_allow_html=True
)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    try:
        input_df.drop("customerID", axis=1, inplace=True, errors='ignore')
        input_df.replace(" ", pd.NA, inplace=True)
        input_df.dropna(inplace=True)
        input_df["TotalCharges"] = input_df["TotalCharges"].astype(float)

        from sklearn.preprocessing import LabelEncoder
        cat_cols = input_df.select_dtypes(include='object').columns
        for col in cat_cols:
            if col != 'Churn':
                input_df[col] = LabelEncoder().fit_transform(input_df[col])

        churn_probs = model.predict_proba(input_df)[:, 1]
        input_df["Churn_Probability"] = churn_probs
        input_df["Predicted_Churn"] = input_df["Churn_Probability"].apply(lambda x: "Yes" if x > 0.5 else "No")

        st.success("✅ Predictions Completed")
        st.write("### 📄 Churn Prediction Results", input_df.head())

        fig, ax = plt.subplots()
        sns.countplot(data=input_df, x="Predicted_Churn", ax=ax, palette="Set2")
        st.pyplot(fig)

        st.sidebar.subheader("🔎 Filter Predictions")
        contract_options = input_df["Contract"].unique().tolist()
        selected_contracts = st.sidebar.multiselect("Select Contract Types", contract_options, default=contract_options)
        filtered_df = input_df[input_df["Contract"].isin(selected_contracts)]

        st.write("### 🔍 Filtered Results")
        st.dataframe(filtered_df)

        st.subheader("💡 Recommendations")
        def generate_recommendation(row):
            if row["Predicted_Churn"] == "Yes":
                recs = []
                if row.get("Contract") == 0:
                    recs.append("Offer yearly contract discount.")
                if row.get("tenure") < 6:
                    recs.append("Send loyalty rewards.")
                if row.get("InternetService") == 1:
                    recs.append("Provide service guarantees.")
                return " | ".join(recs)
            else:
                return "Customer likely to stay."

        filtered_df["Recommendation"] = filtered_df.apply(generate_recommendation, axis=1)
        st.dataframe(filtered_df[["Churn_Probability", "Predicted_Churn", "Recommendation"]].head(10))

        st.subheader("🔍 SHAP Explanation (Feature Impact)")
        shap.initjs()
        sample_input = input_df.drop(columns=["Churn_Probability", "Predicted_Churn"]).head(100)

        @st.cache_resource
        def get_shap_explainer(_model, data_sample):  
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_sample)
            return explainer, shap_values

        explainer, shap_values = get_shap_explainer(model, sample_input)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, sample_input, show=False)  # Don't auto-show
        plt.tight_layout()
        st.pyplot(fig)

        st.download_button("Download Results CSV", data=input_df.to_csv(index=False), file_name="churn_predictions.csv")

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.markdown("---")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
