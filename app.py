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
tabs = ["Upload & Predict", "Visualizations", "Churn Summary", "Chat with AI Support"]
selected_tab = st.sidebar.radio("Navigate", tabs)

# Initialize global DataFrame
if "churn_predictions_df" not in st.session_state:
    st.session_state.churn_predictions_df = pd.DataFrame()
    #churn_predictions_df.columns = churn_predictions_df.columns.str.strip()
    #churn_predictions_df.columns = churn_predictions_df.columns.str.lower()

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

        # If user revisits the tab and file is already in session
    elif "uploaded_file" in st.session_state:
        st.write("### Sample Data of the uploaded file")
        st.dataframe(st.session_state["uploaded_df"].head())

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

        st.markdown("### 🔹 Uploaded data numerical insights", unsafe_allow_html=True)
        view_option = st.radio(
            "Choose how to view the insights:",
            ('Summary Table', 'Box Plots')
        )
        num_cols = user_df.select_dtypes(include=["int", "float"]).columns.tolist()

        if view_option == 'Summary Table':
            #summary_df = user_df[num_cols].describe().T
            #summary_df = summary_df.drop(columns=["count"])
            #st.dataframe(summary_df)

            num_cols = user_df.select_dtypes(include=["int", "float"]).columns.tolist()
            # Remove customerID if it's in num_cols
            num_cols = [col for col in num_cols if col.lower() != "customerid"]
            summary_df = user_df[num_cols].describe().T
            summary_df = summary_df.drop(columns=["count"])
            st.dataframe(summary_df)
               
        else:
            st.write("### Box Plots")
            num_cols = [col for col in num_cols if col.lower() != "customerid"]
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.boxplot(data=user_df, x="Churn", y=col, palette="Set2", ax=ax)
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

elif selected_tab == "Chat with AI Support":
    st.title("🤖 Chat with AI Support")
    st.markdown("The agent will respond based on your churn profile.")

    # ✅ Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    churn_predictions_df = st.session_state.get("churn_predictions_df", None)
    
    if churn_predictions_df is not None and "customerID" in churn_predictions_df.columns:
        churn_predictions_df.columns = churn_predictions_df.columns.str.strip()
        customer_ids = churn_predictions_df["customerID"].unique()
        customer_id = st.selectbox("Select a Customer ID", customer_ids)

        # Clear chat history if customer_id changes
        if "last_customer_id" not in st.session_state:
            st.session_state.last_customer_id = customer_id

        if customer_id != st.session_state.last_customer_id:
            st.session_state.chat_history = []  # Clear chat history
            st.session_state.last_customer_id = customer_id

        customer_data = churn_predictions_df[churn_predictions_df["customerID"] == customer_id]
        
        if not customer_data.empty:
            st.write("📄 Customer Profile:")
            st.dataframe(customer_data.T)

            predicted_churn = customer_data["Churn"].values[0]
            profile_text = customer_data.drop(columns=["customerID", "Churn"]).to_dict(orient="records")[0]

            # ✅ Display chat history on top
            st.subheader("💬 Chat History")
            for sender, msg in st.session_state.chat_history:
                if sender == "You":
                    st.markdown(f"👤 **You:** {msg}")
                else:
                    st.markdown(f"🤖 **Agent:** {msg}")
            
            # ✅ Input box
            #user_input = st.text_input("💬 You (Ask AI Agent about this customer):", placeholder="Type your query")
            user_input = st.text_input("", placeholder="Type your query here and get your answer from AI Agent about this customer")
            
            if user_input:
                with st.spinner("Generating response..."):
                    try:
                        customer_data_dict = customer_data.iloc[0].drop(["customerID"]).to_dict()
                        reply = generate_response(customer_data_dict, user_input)

                         # ✅ Save chat history (new messages on top)
                        st.session_state.chat_history.insert(0, ("Agent", reply))
                        st.session_state.chat_history.insert(0, ("You", user_input))

                        #if you still want the latest reply to appear immediately below the input box, just add this line after inserting into chat_history:
                        st.markdown(f"**Agent:** {reply}")

                    except Exception as e:
                        st.error(f"❌ LLM Error: {str(e)}")
        else:
            st.warning("Selected customer ID not found in uploaded data.")
    else:
        st.info("📤 Please upload and predict data first in the 'Upload & Predict' tab.")
