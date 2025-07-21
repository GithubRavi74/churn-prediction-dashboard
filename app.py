import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cloudpickle
from sklearn.metrics import classification_report, confusion_matrix

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
    uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("### Sample Data", input_df.head())

        # Predict churn
        predictions = pipeline.predict(input_df)
        input_df["churn_prediction"] = np.where(predictions == 1, "Churn", "No Churn")

        st.write("### Prediction Results")
        st.dataframe(input_df)

        # Save to session state
        st.session_state.churn_predictions_df = input_df

elif selected_tab == "Visualizations":
    st.subheader("📊 Churn Distribution Visualization")

    if not st.session_state.churn_predictions_df.empty:
        user_df = st.session_state.churn_predictions_df

        st.markdown("### 🔹 Churn Count Plot", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots()
        sns.countplot(data=user_df, x="churn_prediction", palette="coolwarm", ax=ax1)
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

        churn_count = df["churn_prediction"].value_counts()
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

    st.markdown("Ask any question or type your concerns as a customer. The agent will respond based on your churn profile.")

    if st.session_state.churn_predictions_df.empty:
        st.warning("Please upload and predict data in the 'Upload & Predict' tab first.")
    else:
        churn_predictions_df = st.session_state.churn_predictions_df
        #st.write("Available columns:", churn_predictions_df.columns.tolist())
        #st.dataframe(churn_predictions_df.head())
        customer_id = st.selectbox("Select a Customer ID", churn_predictions_df["customerID"].unique())
        customer_row = churn_predictions_df[churn_predictions_df["customerID"] == customer_id].iloc[0]

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("You:", key="chat_input")

        if user_input:
            st.session_state.chat_history.append(("user", user_input))

            if customer_row["churn_prediction"] == "Churn":
                if "price" in user_input.lower():
                    agent_reply = "It seems pricing might be a concern. Would a discount or better plan suit you?"
                elif "why" in user_input.lower() or "reason" in user_input.lower():
                    agent_reply = f"Our model identified contract type and tenure as churn indicators. We’d love to keep you with a new offer."
                else:
                    agent_reply = "We understand your concerns. Let me check how we can retain you with a personalized offer."
            else:
                agent_reply = "You're a valued customer with no signs of churn. Is there anything else I can help you with?"

            st.session_state.chat_history.append(("agent", agent_reply))

        for sender, msg in st.session_state.chat_history:
            if sender == "user":
                st.markdown(f"👤 **You**: {msg}")
            else:
                st.markdown(f"🤖 **Agent**: {msg}")
