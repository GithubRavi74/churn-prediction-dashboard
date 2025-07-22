import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables (useful if you're running locally with a .env file)
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response(customer_data_dict, user_message):
    """
    Generates a response from the AI agent based on customer data and user message.

    Args:
        customer_data_dict (dict): Dictionary of customer attributes.
        user_message (str): Message/question entered by the user.

    Returns:
        str: AI-generated reply.
    """
    prompt = f"""You are a customer retention agent. You are given this customer's data:
{customer_data_dict}

The user asked:
"{user_message}"

Respond as a helpful AI agent trying to retain the customer. Keep it friendly, clear, and informative."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful customer retention AI agent."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
