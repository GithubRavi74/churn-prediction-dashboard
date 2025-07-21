import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_response(customer_data: dict, user_message: str) -> str:
    customer_info = ", ".join([f"{k}: {v}" for k, v in customer_data.items()])
    prompt = f"""
You are a customer retention assistant for a telecom company. Based on the customer's profile and their concern, give a helpful and friendly reply.

Customer Details: {customer_info}
Customer Message: {user_message}

Your Response:
"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if you're on Pro
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message["content"].strip()

    except Exception as e:
        return f"Error generating response: {e}"
