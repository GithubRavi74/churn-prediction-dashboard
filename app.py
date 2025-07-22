import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cloudpickle
from sklearn.metrics import classification_report, confusion_matrix
from churn_agent_llm import generate_response


# Load the trained pipeline
with open("churn_pipeline.pkl", "rb") as file:
    pipeline = cloudpickle.load(file)

# App title
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 NTTIS AI SOLUTION - Customer Churn Prediction Dashboard")

# Tabs
tabs = ["Upload & Predict", "Visualizations", "Churn Summary", "Chat with Agent"]
selected_tab = st.sidebar.radio("Navigate", tabs)

# Initialize global DataFrame
if "churn_predictions_df" not in st.session_state:
    st.session_state.churn_predictions_df = pd.DataFrame()
    churn_predictions_df.columns = churn_predictions_df.columns.str.strip()
    churn_predictions_df.columns = churn_predictions_df.columns.str.lower()

if selected_tab == "Upload & Predict":
    st.subheader("📤 Upload Customer Data (CSV)")
    #####################
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        user_df.columns = user_df.columns.map(str).str.strip()
        st.session_state["uploaded_df"] = user_df
        st.write("Sample of uploaded data:")
        st.dataframe(user_df.head())
        
        # Predict churn
        churn_predictions = pipeline.predict(user_df)
        churn_proba = pipeline.predict_proba(user_df)[:, 1]
        user_df["Churn"] = churn_predictions
        user_df["Churn_Probability"] = churn_proba
        st.session_state["churn_predictions_df"] = user_df
        st.write("Predictions:")
        st.dataframe(user_df[["customerID", "Churn", "Churn_Probability"]])
        # Save to session state
        st.session_state.churn_predictions_df = user_df
elif selected_tab == "Visualizations":
    st.subheader("📊 Churn Distribution Visualization")

    if not st.session_state.churn_predictions_df.empty:
        user_df = st.session_state.churn_predictions_df

        st.markdown("### 🔹 Churn Count Plot", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots()
        sns.countplot(data=user_df, x="Churn", palette="coolwarm", ax=ax1)
        ax1.set_title("Customer Churn Distribution", fontsize=14)
        st.pyplot(fig1)

        st.markdown("---")  # Line break

        st.markdown("### 🔹 Numerical Feature Analysis", unsafe_allow_html=True)
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
                sns.boxplot(data=user_df, x="churn_prediction", y=col, palette="Set2", ax=ax)
                ax.set_title(f"{col} by Churn Status", fontsize=12)
                st.pyplot(fig)

    else:
        st.warning("Please upload and predict data in the 'Upload & Predict' tab first.")

elif selected_tab == "Churn Summary":
    st.subheader("📌 Churn Insights Summary")

    if not st.session_state.churn_predictions_df.empty:
        df = st.session_state.churn_predictions_df

        churn_count = df["Churn"].value_counts()
        st.write("### Churn Counts", churn_count)

        st.write("### Churn Percentage")
        churn_pct = churn_count / len(df) * 100
        st.dataframe(churn_pct)

        if "Churn" in churn_count:
            st.success(f"🚨 {churn_count['Churn']} customers predicted to churn ({churn_pct['Churn']:.2f}%)")
        else:
            st.info("✅ No customers predicted to churn.")
    else:
        st.warning("Please upload and predict data in the 'Upload & Predict' tab first.")

elif selected_tab == "Chat with Agent":
    st.title("🤖 Chat with Agent")
    st.markdown("The agent will respond based on your churn profile.")

    if churn_predictions_df is not None and "customerID" in churn_predictions_df.columns:
        churn_predictions_df.columns = churn_predictions_df.columns.str.strip()
        customer_ids = churn_predictions_df["customerID"].unique()
        customer_id = st.selectbox("Select a Customer ID", customer_ids)

        customer_data = churn_predictions_df[churn_predictions_df["customerID"] == customer_id]
        
        if not customer_data.empty:
            st.write("📄 Customer Profile:")
            st.dataframe(customer_data.T)

            predicted_churn = customer_data["Churn"].values[0]
            profile_text = customer_data.drop(columns=["customerID", "Churn"]).to_dict(orient="records")[0]

            user_input = st.text_input("💬 You (Ask the Agent about this customer):", placeholder="Why might this customer churn?")
            
            if user_input:
                with st.spinner("Generating response..."):
                    try:
                        import openai
                        openai.api_key = st.secrets["OPENAI_API_KEY"]

                        system_prompt = (
                            "You are a customer retention agent. Use the customer's profile and churn prediction to offer helpful suggestions or answer queries. "
                            "Be empathetic and analytical."
                        )

                        formatted_profile = "\n".join([f"{k}: {v}" for k, v in profile_text.items()])
                        full_prompt = f"""
Customer ID: {customer_id}
Churn Prediction: {predicted_churn}
Customer Profile:
{formatted_profile}

User Query: {user_input}
                        """

                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": full_prompt}
                            ],
                            temperature=0.7,
                            max_tokens=500,
                        )

                        reply = response['choices'][0]['message']['content']
                        st.markdown(f"**Agent:** {reply}")

                    except Exception as e:
                        st.error(f"❌ LLM Error: {str(e)}")
        else:
            st.warning("Selected customer ID not found in uploaded data.")
    else:
        st.info("📤 Please upload and predict data first in the 'Upload & Predict' tab.")
