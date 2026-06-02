from openai import OpenAI
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

client = OpenAI(api_key=api_key)

# Chat memory
messages = [
    {
        "role": "system",
        "content": """
You are SupportSmart AI, an AI customer service chatbot for a large e-commerce company.

Your job:
- Help customers with product enquiries
- Help with order tracking questions
- Help with delivery issues
- Help with refund and return questions
- Help with complaints
- Give clear, polite and professional customer support

Important behaviour:
- Be friendly, calm and professional.
- Ask for missing details when needed.
- Do not invent order details, refund status, delivery dates, prices or company policies.
- If information is not provided, say you do not have access to that information.
- For complex, emotional, angry, legal, refund dispute, payment issue, account security, privacy, or repeated unresolved problems, escalate to a human agent.
- Always protect customer privacy.
- Do not ask for full card numbers, passwords, or sensitive personal information.
- Keep answers short, helpful and easy to understand.

Hybrid Human-AI Support Rule:
If the customer issue is complex or sensitive, clearly say:
"I’ll escalate this to a human support agent so they can review it properly."

After helping, ask:
"Did this answer your question, or would you like me to connect you with a human agent?"
"""
    }
]


def classify_query(user_input):
    """
    Simple rule-based classifier.
    This helps the chatbot decide whether to answer or escalate.
    """

    complex_keywords = [
        "angry", "furious", "complaint", "refund rejected", "not refunded",
        "legal", "lawsuit", "fraud", "scam", "stolen", "hacked",
        "payment failed", "charged twice", "wrong charge",
        "privacy", "personal data", "manager", "human agent",
        "cancelled but charged", "damaged", "missing item",
        "not delivered", "escalate"
    ]

    simple_keywords = [
        "track", "delivery", "return policy", "opening hours",
        "product", "size", "available", "shipping", "order status"
    ]

    text = user_input.lower()

    for word in complex_keywords:
        if word in text:
            return "complex"

    for word in simple_keywords:
        if word in text:
            return "simple"

    return "unknown"


def chat_with_support_bot(user_input):
    query_type = classify_query(user_input)

    if query_type == "complex":
        escalation_note = """
The user query appears complex or sensitive.
You should respond empathetically and escalate to a human support agent.
Do not try to fully resolve the issue if account/order/payment/private details are required.
"""
    elif query_type == "simple":
        escalation_note = """
The user query appears simple.
You may answer normally, but do not invent details.
Ask for order number or missing information if needed.
"""
    else:
        escalation_note = """
The query type is unclear.
Ask a clarifying question before giving a final answer.
"""

    messages.append(
        {
            "role": "user",
            "content": f"""
Customer message:
{user_input}

Query classification:
{query_type}

Instruction:
{escalation_note}
"""
        }
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages
    )

    bot_reply = response.output_text

    messages.append(
        {
            "role": "assistant",
            "content": bot_reply
        }
    )

    return bot_reply


print("===== SupportSmart AI Chatbot =====")
print("AI customer service chatbot for a large e-commerce company.")
print("Type 'exit' to stop.\n")

while True:
    user_input = input("Customer: ")

    if user_input.lower() in ["exit", "quit"]:
        print("SupportSmart AI: Thank you for contacting support. Goodbye!")
        break

    answer = chat_with_support_bot(user_input)

    print("\nSupportSmart AI:")
    print(answer)
    print()