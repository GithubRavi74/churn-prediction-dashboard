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
    st.markdown("The agent will respond based on your churn profile.")

    # Ensure previous predictions are available
    if st.session_state.churn_predictions_df.empty:
        st.warning("Please upload and predict data in the 'Upload & Predict' tab first.")
    else:
        churn_predictions_df = st.session_state.churn_predictions_df.copy()

        # Ensure columns are clean
        churn_predictions_df.columns = churn_predictions_df.columns.map(str).str.strip()
        churn_predictions_df["customerID"] = churn_predictions_df["customerID"].astype(str).str.strip()
       
        # Step 1: Select customer
        customer_id = st.selectbox("Select a Customer ID", churn_predictions_df["customerID"].unique())
        customer_id = str(customer_id).strip()
        customer_data = churn_predictions_df[churn_predictions_df["customerID"] == customer_id]

        # Step 2: Show churn prediction
        if not customer_data.empty:
            predicted_churn = customer_data["Churn"].values[0]
            st.write(f"**Churn Prediction for {customer_id}:** `{predicted_churn}`")
        else:
            st.error("No customer data found. Please check the selected customer ID.")
            predicted_churn = None

        # Step 3: Initialize chat history if not exists
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Step 4: User input
        user_input = st.text_input("You", placeholder="Type your issue or concern here...")

        # Step 5: Agent response
        def retention_response(message):
            message = message.lower()
            if "cancel" in message:
                return "We’re sorry to hear that. Can we offer a 10% discount to retain you?"
            elif "price" in message or "bill" in message:
                return "We understand billing concerns. We can offer a flexible plan with no extra charges."
            elif "speed" in message:
                return "We’re actively upgrading speed in your area. We can send a free technician visit."
            elif "service" in message or "issue" in message:
                return "We're sorry for the inconvenience. Our support team can prioritize your case now."
            elif "switch" in message or "competitor" in message:
                return "Loyal customers get 2 months free! Would that help you stay with us?"
            else:
                return "Thank you for reaching out. We’ll have our agent get in touch shortly."

        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            # commenting the hard coded approach so as to use LLM
            #if predicted_churn == "Churn":
            #    reply = retention_response(user_input)
            #elif predicted_churn == "No Churn":
            #    reply = "You're a valued customer with no signs of churn. Is there anything else I can help you with?"
            #else:
            #    reply = "Sorry, we could not determine the churn prediction."

              
        # Step 6: Display chat history
        st.markdown("### 💬 Chat History")

        #for sender, msg in st.session_state.chat_history:
            #if sender == "user":
                #st.markdown(f"👤 **You:** {msg}")
            #else:
                #st.markdown(f"🤖 **Agent:** {msg}")

            st.session_state.chat_history.append(("agent", reply))
            if st.button("Ask Agent"):
                customer_data_dict = customer_data.iloc[0].to_dict()
                reply = generate_response(customer_data_dict, user_message)
                st.markdown(f"**Agent:** {reply}")


        
        
