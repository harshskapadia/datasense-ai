import os
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 1. LOAD API KEY
# =========================================================

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Please check your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)


# =========================================================
# 2. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="SupportSmart AI",
    page_icon="💬",
    layout="wide"
)


# =========================================================
# 3. CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 44px;
        font-weight: 800;
        text-align: center;
        color: #1f2937;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: #f9fafb;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .status-simple {
        background-color: #dcfce7;
        color: #166534;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }

    .status-complex {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }

    .status-unclear {
        background-color: #fef3c7;
        color: #92400e;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: 600;
        display: inline-block;
    }

    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 13px;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 4. TITLE
# =========================================================

st.markdown(
    '<div class="main-title">💬 SupportSmart AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Hybrid AI Customer Service Chatbot for Large E-Commerce Companies</div>',
    unsafe_allow_html=True
)


# =========================================================
# 5. SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_log" not in st.session_state:
    st.session_state.conversation_log = []

if "escalation_log" not in st.session_state:
    st.session_state.escalation_log = []

if "query_count" not in st.session_state:
    st.session_state.query_count = 0

if "simple_count" not in st.session_state:
    st.session_state.simple_count = 0

if "complex_count" not in st.session_state:
    st.session_state.complex_count = 0

if "last_classification" not in st.session_state:
    st.session_state.last_classification = "None"

if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = "None"

if "last_escalation" not in st.session_state:
    st.session_state.last_escalation = "No"


# =========================================================
# 6. COMPANY KNOWLEDGE BASE
# =========================================================

COMPANY_NAME = "MegaMart Online"

FAQ_KNOWLEDGE = """
Company: MegaMart Online

General Support Policies:
- Customers can ask about product information, delivery, returns, refunds, damaged items, missing items, and account support.
- The chatbot does not have access to real customer accounts, live order tracking, payment systems, or private customer records.
- For order-specific issues, the chatbot should ask for an order number but must not invent order status.
- For payment, refund disputes, account security, privacy concerns, legal complaints, or angry customers, the chatbot should escalate to a human support agent.
- The chatbot must not ask for passwords, full card numbers, bank details, or sensitive personal information.
- The chatbot should be polite, calm, professional, and transparent.
- If information is not available, the chatbot should clearly say it does not have access to that information.
"""


# =========================================================
# 7. QUERY CLASSIFICATION
# =========================================================

def classify_query(user_message):
    text = user_message.lower()

    complex_keywords = [
        "angry",
        "furious",
        "complaint",
        "manager",
        "human",
        "agent",
        "refund rejected",
        "not refunded",
        "charged twice",
        "wrong charge",
        "payment",
        "card",
        "fraud",
        "scam",
        "hacked",
        "stolen",
        "privacy",
        "personal data",
        "legal",
        "lawyer",
        "lawsuit",
        "damaged",
        "missing item",
        "not delivered",
        "late for weeks",
        "cancelled but charged",
        "escalate"
    ]

    simple_keywords = [
        "track",
        "order",
        "delivery",
        "shipping",
        "return policy",
        "refund policy",
        "product",
        "size",
        "available",
        "price",
        "warranty",
        "how long",
        "where is",
        "exchange"
    ]

    angry_words = [
        "angry",
        "furious",
        "annoyed",
        "frustrated",
        "terrible",
        "worst",
        "useless",
        "bad service"
    ]

    emotion = "Neutral"

    for word in angry_words:
        if word in text:
            emotion = "Frustrated / Angry"
            break

    for word in complex_keywords:
        if word in text:
            return {
                "query_type": "Complex / Sensitive Issue",
                "complexity": "Complex",
                "emotion": emotion,
                "escalation_needed": "Yes",
                "reason": "The message includes a sensitive, emotional, payment, refund, privacy, complaint, or unresolved issue."
            }

    for word in simple_keywords:
        if word in text:
            return {
                "query_type": "General Customer Support",
                "complexity": "Simple",
                "emotion": emotion,
                "escalation_needed": "No",
                "reason": "The message appears to be a routine customer support question."
            }

    return {
        "query_type": "Unclear / Needs Clarification",
        "complexity": "Unclear",
        "emotion": emotion,
        "escalation_needed": "Maybe",
        "reason": "The message does not clearly match a known support category, so clarification is needed."
    }


# =========================================================
# 8. AI RESPONSE FUNCTION
# =========================================================

def generate_support_response(user_message, classification):
    system_prompt = f"""
You are SupportSmart AI, a customer service chatbot for {COMPANY_NAME}.

You are designed based on a hybrid human-AI customer support framework.

Your goals:
- Help customers with simple customer support issues.
- Protect customer trust by being accurate and transparent.
- Avoid hallucinated or fake information.
- Escalate complex, emotional, sensitive, refund dispute, payment, privacy, legal, security, or repeated unresolved issues to a human agent.
- Improve customer satisfaction by giving clear and helpful responses.

Company Knowledge:
{FAQ_KNOWLEDGE}

Customer Query Classification:
Query Type: {classification["query_type"]}
Complexity: {classification["complexity"]}
Customer Emotion: {classification["emotion"]}
Escalation Needed: {classification["escalation_needed"]}
Reason: {classification["reason"]}

Rules:
- Be polite, calm, and professional.
- Do not invent order status, delivery dates, refund approvals, prices, or company policies.
- If order/account-specific information is needed, say you do not have access to live account data.
- Do not ask for passwords, full card numbers, bank details, or sensitive private data.
- If escalation is needed, clearly say you will connect the customer with a human support agent.
- For simple issues, answer clearly and ask for missing details if needed.
- End with a satisfaction/trust check question.

Response style:
- Short but helpful.
- Use simple language.
- Use bullet points only if useful.
"""

    messages_for_api = [
        {"role": "system", "content": system_prompt}
    ]

    for msg in st.session_state.messages[-8:]:
        messages_for_api.append(msg)

    messages_for_api.append(
        {"role": "user", "content": user_message}
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages_for_api
    )

    return response.output_text


# =========================================================
# 9. SIDEBAR
# =========================================================

st.sidebar.title("📌 SupportSmart AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Chatbot",
        "Dashboard",
        "Escalation Log",
        "Project Summary"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Research-Based Design")
st.sidebar.write("Technical Performance")
st.sidebar.write("Customer Trust")
st.sidebar.write("Hybrid Human-AI Support")
st.sidebar.write("Ethical Reliability")
st.sidebar.write("Human Escalation")

st.sidebar.markdown("---")
st.sidebar.info(
    "This chatbot is designed for online retail/e-commerce support scenarios such as delivery, refunds, complaints, product enquiries, and post-sale support."
)


# =========================================================
# 10. PAGE: CHATBOT
# =========================================================

if page == "Chatbot":
    st.header("Customer Support Chatbot")

    st.write(
        "This chatbot handles routine customer queries and escalates complex or sensitive issues to a human agent."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🛒 Product enquiry"):
            st.session_state.example_prompt = "I want to know if this product is available in size medium."

    with col2:
        if st.button("🚚 Delivery issue"):
            st.session_state.example_prompt = "Where is my order? It was supposed to arrive yesterday."

    with col3:
        if st.button("⚠️ Refund complaint"):
            st.session_state.example_prompt = "I am very angry. My refund was rejected and I want to speak to someone."

    if "example_prompt" not in st.session_state:
        st.session_state.example_prompt = ""

    user_input = st.chat_input("Type your customer message here...")

    if st.session_state.example_prompt:
        user_input = st.session_state.example_prompt
        st.session_state.example_prompt = ""

    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])

        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])

    if user_input:
        classification = classify_query(user_input)

        st.session_state.query_count += 1
        st.session_state.last_classification = classification["complexity"]
        st.session_state.last_emotion = classification["emotion"]
        st.session_state.last_escalation = classification["escalation_needed"]

        if classification["complexity"] == "Simple":
            st.session_state.simple_count += 1
        elif classification["complexity"] == "Complex":
            st.session_state.complex_count += 1

        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.spinner("SupportSmart AI is thinking..."):
            bot_reply = generate_support_response(user_input, classification)

        with st.chat_message("assistant"):
            st.write(bot_reply)

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_reply}
        )

        log_entry = {
            "Customer Message": user_input,
            "Query Type": classification["query_type"],
            "Complexity": classification["complexity"],
            "Emotion": classification["emotion"],
            "Escalation Needed": classification["escalation_needed"],
            "Reason": classification["reason"],
            "Bot Response": bot_reply
        }

        st.session_state.conversation_log.append(log_entry)

        if classification["escalation_needed"] == "Yes":
            st.session_state.escalation_log.append(log_entry)

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Last Complexity", st.session_state.last_classification)

    with col_b:
        st.metric("Last Emotion", st.session_state.last_emotion)

    with col_c:
        st.metric("Escalation Needed", st.session_state.last_escalation)

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.conversation_log = []
        st.session_state.escalation_log = []
        st.session_state.query_count = 0
        st.session_state.simple_count = 0
        st.session_state.complex_count = 0
        st.session_state.last_classification = "None"
        st.session_state.last_emotion = "None"
        st.session_state.last_escalation = "No"
        st.rerun()


# =========================================================
# 11. PAGE: DASHBOARD
# =========================================================

elif page == "Dashboard":
    st.header("Chatbot Performance Dashboard")

    st.write(
        "This dashboard shows how the chatbot classifies support requests and when it escalates issues."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", st.session_state.query_count)

    with col2:
        st.metric("Simple Queries", st.session_state.simple_count)

    with col3:
        st.metric("Complex Queries", st.session_state.complex_count)

    with col4:
        st.metric("Escalations", len(st.session_state.escalation_log))

    st.subheader("Conversation Log")

    if st.session_state.conversation_log:
        log_df = pd.DataFrame(st.session_state.conversation_log)
        st.dataframe(log_df, use_container_width=True)

        csv_data = log_df.to_csv(index=False)

        st.download_button(
            label="Download Conversation Log",
            data=csv_data,
            file_name="supportsmart_conversation_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No conversation data yet. Go to the Chatbot page and test some messages.")


# =========================================================
# 12. PAGE: ESCALATION LOG
# =========================================================

elif page == "Escalation Log":
    st.header("Human Escalation Log")

    st.write(
        "Complex, emotional, payment, privacy, refund dispute, or security-related queries are logged here for human support."
    )

    if st.session_state.escalation_log:
        escalation_df = pd.DataFrame(st.session_state.escalation_log)

        st.dataframe(escalation_df, use_container_width=True)

        csv_data = escalation_df.to_csv(index=False)

        st.download_button(
            label="Download Escalation Log",
            data=csv_data,
            file_name="supportsmart_escalation_log.csv",
            mime="text/csv"
        )
    else:
        st.success("No escalated cases yet.")


# =========================================================
# 13. PAGE: PROJECT SUMMARY
# =========================================================

elif page == "Project Summary":
    st.header("Project Summary")

    summary = """
SupportSmart AI is a hybrid AI customer service chatbot designed for large e-commerce companies.

The chatbot is based on a research proposal evaluating the effectiveness of NLP-based AI chatbots in customer support. It focuses on three major dimensions:
- Technical performance
- Customer trust and satisfaction
- Hybrid human-AI support

Main Features:
- Customer support chatbot interface
- Query classification
- Emotion/sensitivity detection
- Simple query handling
- Human escalation for complex or sensitive issues
- Conversation logging
- Escalation logging
- Dashboard for support analysis
- Downloadable CSV logs

Use Cases:
- Product enquiries
- Delivery issues
- Refund and return questions
- Complaints
- Payment/security concerns
- Privacy-related issues

Technology Stack:
- Python
- Streamlit
- OpenAI API
- pandas
- python-dotenv
"""

    st.markdown(summary)

    st.subheader("LinkedIn Caption Draft")

    linkedin_caption = """
I built SupportSmart AI, a hybrid AI customer service chatbot designed for large e-commerce support environments.

The project is based on my research proposal about evaluating NLP-based AI chatbots in customer support. It focuses on technical performance, customer trust, satisfaction, ethical reliability, and hybrid human-AI support.

The chatbot can classify customer queries, respond to routine issues, detect complex or sensitive cases, escalate them to human agents, log conversations, and provide a dashboard for analysing chatbot performance.

Tech stack: Python, Streamlit, OpenAI API, pandas, and python-dotenv.
"""

    st.text_area("Copy LinkedIn caption:", linkedin_caption, height=220)

    st.download_button(
        label="Download Project Summary",
        data=summary + "\n\nLinkedIn Caption:\n" + linkedin_caption,
        file_name="supportsmart_project_summary.txt",
        mime="text/plain"
    )


# =========================================================
# 14. FOOTER
# =========================================================

st.markdown(
    '<div class="footer">Built with Python, Streamlit, OpenAI API and pandas.</div>',
    unsafe_allow_html=True
)