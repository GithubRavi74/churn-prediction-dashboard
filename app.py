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
st.title("📉 NTTIS AI SOLUTION - CCPD-  Customer Churn Prediction Dashboard")

# ✅ Clear pending input if flagged
if "pending_clear_input" in st.session_state:
    if st.session_state.pending_clear_input in st.session_state:
        st.session_state[st.session_state.pending_clear_input] = ""
    del st.session_state.pending_clear_input

# Tabs
tabs = ["Upload & Predict", "Visualizations", "Churn Summary", "Chat with AI Support"]
selected_tab = st.sidebar.radio("Navigate", tabs)

# Initialize global DataFrame
if "churn_predictions_df" not in st.session_state:
    st.session_state.churn_predictions_df = pd.DataFrame()

if selected_tab == "Upload & Predict":
    st.subheader("📤 Upload Customer Data (CSV)")
    uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

    if uploaded_file:
        user_df = pd.read_csv(uploaded_file)
        user_df.columns = user_df.columns.map(str).str.strip()
        st.session_state["uploaded_df"] = user_df
        st.write("Sample of uploaded data:")
        st.dataframe(user_df.head())

        churn_predictions = pipeline.predict(user_df)
        churn_proba = pipeline.predict_proba(user_df)[:, 1]
        user_df["Churn"] = churn_predictions
        user_df["Churn_Probability"] = churn_proba
        st.session_state["churn_predictions_df"] = user_df

        st.write("Predictions:")
        st.dataframe(user_df[["customerID", "Churn", "Churn_Probability"]])

elif selected_tab == "Visualizations":
    st.subheader("📊 Churn Distribution Visualization")

    if not st.session_state.churn_predictions_df.empty:
        user_df = st.session_state.churn_predictions_df

        st.markdown("### 🔹 Churn Count Plot", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots()
        sns.countplot(data=user_df, x="Churn", palette="coolwarm", ax=ax1)
        ax1.set_title("Customer Churn Distribution", fontsize=14)
        st.pyplot(fig1)

        st.markdown("---")
        st.markdown("### 🔹 Uploaded data numerical insights", unsafe_allow_html=True)

        view_option = st.radio(
            "Choose how to view the insights:",
            ("Summary Table", "Box Plots"),
        )

        num_cols = user_df.select_dtypes(include=["int", "float"]).columns.tolist()
        num_cols = [col for col in num_cols if col.lower() != "customerid"]

        if view_option == "Summary Table":
            summary_df = user_df[num_cols].describe().T.drop(columns=["count"])
            st.dataframe(summary_df)
        else:
            st.write("### Box Plots")
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.boxplot(data=user_df, x="Churn", y=col, palette="Set2", ax=ax)
                ax.set_title(f"{col} by Churn Status", fontsize=12)
                st.pyplot(fig)

elif selected_tab == "Churn Summary":
    st.subheader("📌 Churn Insights Summary")

    if not st.session_state.churn_predictions_df.empty:
        df = st.session_state.churn_predictions_df
        churn_count = df["Churn"].value_counts()

        st.write("### Churn Counts", churn_count)
        churn_pct = churn_count / len(df) * 100
        st.dataframe(churn_pct)

        if "Churn" in churn_count:
            st.success(f"🚨 {churn_count['Churn']} customers predicted to churn ({churn_pct['Churn']:.2f}%)")
        else:
            st.info("✅ No customers predicted to churn.")
    else:
        st.warning("Please upload and predict data first in the 'Upload & Predict' tab.")

elif selected_tab == "Chat with AI Support":
    st.title("🤖 Chat with AI Support")
    st.markdown("The agent will respond based on your churn profile.")

    churn_predictions_df = st.session_state.get("churn_predictions_df", None)

    if churn_predictions_df is not None and "customerID" in churn_predictions_df.columns:
        churn_predictions_df.columns = churn_predictions_df.columns.str.strip()
        customer_ids = churn_predictions_df["customerID"].unique()
        customer_id = st.selectbox("Select a Customer ID", customer_ids)

        # Ensure chat history dictionary
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = {}

        if "last_customer_id" not in st.session_state:
            st.session_state.last_customer_id = customer_id

        # Reset history if new customer selected
        if customer_id != st.session_state.last_customer_id:
            st.session_state.last_customer_id = customer_id
            if customer_id not in st.session_state.chat_history:
                st.session_state.chat_history[customer_id] = []

        if customer_id not in st.session_state.chat_history:
            st.session_state.chat_history[customer_id] = []

        customer_data = churn_predictions_df[churn_predictions_df["customerID"] == customer_id]

        if not customer_data.empty:
            st.write("📄 Customer Profile:")
            st.dataframe(customer_data.T)

            chat_placeholder = st.empty()

            def render_chat():
                with chat_placeholder.container():
                    for sender, msg in st.session_state.chat_history[customer_id]:
                        if sender == "user":
                            st.markdown(f"👤 **You:** {msg}")
                        else:
                            st.markdown(f"🤖 **Agent:** {msg}")

            render_chat()

            user_input_key = f"chat_input_{customer_id}"

            user_input = st.text_input(
                "💬 You (Ask the AI Agent about this customer):",
                placeholder="Type your query here...",
                key=user_input_key
            )

            # ✅ Clear Chat Button below input
            if st.button("🗑️ Clear Chat for this Customer"):
                st.session_state.chat_history[customer_id] = []
                st.session_state.pending_clear_input = user_input_key
                st.rerun()

            if user_input:
                with st.spinner("Generating response..."):
                    try:
                        customer_data_dict = customer_data.iloc[0].drop(["customerID"]).fillna("N/A").to_dict()
                        reply = generate_response(customer_data_dict, user_input)

                        st.session_state.chat_history[customer_id].append(("user", user_input))
                        st.session_state.chat_history[customer_id].append(("agent", reply))

                        render_chat()

                        # ✅ Mark input for clearing after rerun
                        st.session_state.pending_clear_input = user_input_key
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ LLM Error: {str(e)}")

        else:
            st.warning("Selected customer ID not found in uploaded data.")
    else:
        st.info("📤 Please upload and predict data first in the 'Upload & Predict' tab.")
