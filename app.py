import os
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 1. LOAD OPENAI API KEY
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
    page_title="DataSense AI",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# 3. CUSTOM UI
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
        margin-bottom: 35px;
    }

    .info-box {
        background-color: #f9fafb;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 20px;
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

st.markdown('<div class="main-title">📊 DataSense AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-Powered CSV Data Analysis Assistant</div>',
    unsafe_allow_html=True
)


# =========================================================
# 4. SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = ""

if "data_quality_report" not in st.session_state:
    st.session_state.data_quality_report = ""

if "question_answer" not in st.session_state:
    st.session_state.question_answer = ""


# =========================================================
# 5. HELPER FUNCTIONS
# =========================================================

def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        return None


def get_dataset_summary(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_text = buffer.getvalue()

    summary = f"""
DATASET SHAPE
Rows: {df.shape[0]}
Columns: {df.shape[1]}

COLUMN NAMES
{list(df.columns)}

DATA TYPES
{df.dtypes.to_string()}

MISSING VALUES
{df.isnull().sum().to_string()}

DUPLICATE ROWS
{df.duplicated().sum()}

UNIQUE VALUES PER COLUMN
{df.nunique().to_string()}

SUMMARY STATISTICS
{df.describe(include='all').to_string()}

DATAFRAME INFO
{info_text}
"""
    return summary


def generate_ai_insights(df):
    dataset_summary = get_dataset_summary(df)

    prompt = f"""
You are DataSense AI, a data analysis assistant.

Analyse the CSV dataset summary below.

Rules:
- Do not invent facts.
- Only use what is shown in the dataset summary.
- Use simple but professional language.
- Make it useful for a student LinkedIn portfolio project.
- Give practical insights, not generic comments.

Return the answer using these headings:

1. Dataset Overview
2. Key Columns and Their Meaning
3. Data Quality Issues
4. Missing Value Observations
5. Duplicate Row Observations
6. Important Patterns or Possible Trends
7. Suggested Charts
8. Possible Business or Practical Insights
9. Recommended Next Steps

DATASET SUMMARY:
{dataset_summary}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text


def generate_data_quality_report(df):
    dataset_summary = get_dataset_summary(df)

    prompt = f"""
You are DataSense AI, a data quality analyst.

Create a data quality report for the dataset below.

Rules:
- Do not invent information.
- Focus on practical cleaning and preparation steps.
- Use clear headings and bullet points.

Return:

1. Overall Data Quality Score out of 100
2. Main Data Quality Issues
3. Missing Value Handling Suggestions
4. Duplicate Handling Suggestions
5. Column Type Issues
6. Columns That May Need Cleaning
7. Recommended Cleaning Steps
8. Ready for Analysis? Yes/No with reason

DATASET SUMMARY:
{dataset_summary}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text


def answer_dataset_question(df, question):
    dataset_summary = get_dataset_summary(df)

    sample_rows = df.head(10).to_string()

    prompt = f"""
You are DataSense AI, a CSV data assistant.

The user asked a question about their dataset.

Rules:
- Answer only using the dataset summary and sample rows.
- If the answer cannot be confirmed from the provided data, say so.
- Keep the answer clear and practical.

USER QUESTION:
{question}

DATASET SUMMARY:
{dataset_summary}

SAMPLE ROWS:
{sample_rows}
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text


def create_full_report(df):
    report = f"""
DATASENSE AI DATA ANALYSIS REPORT

DATASET BASIC INFORMATION
Rows: {df.shape[0]}
Columns: {df.shape[1]}

COLUMN LIST
{list(df.columns)}

MISSING VALUES
{df.isnull().sum().to_string()}

DUPLICATE ROWS
{df.duplicated().sum()}

SUMMARY STATISTICS
{df.describe(include='all').to_string()}

AI INSIGHTS
{st.session_state.ai_insights if st.session_state.ai_insights else "AI insights not generated yet."}

DATA QUALITY REPORT
{st.session_state.data_quality_report if st.session_state.data_quality_report else "Data quality report not generated yet."}
"""
    return report


# =========================================================
# 6. SIDEBAR
# =========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Choose Section",
    [
        "Upload Dataset",
        "Dataset Overview",
        "Data Quality",
        "Visualisations",
        "AI Insights",
        "Ask DataSense AI",
        "Download Report",
        "Project Summary"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Stack")
st.sidebar.write("Python")
st.sidebar.write("Streamlit")
st.sidebar.write("Pandas")
st.sidebar.write("Matplotlib")
st.sidebar.write("OpenAI API")

st.sidebar.markdown("---")
st.sidebar.info(
    "This version works with almost any CSV file and focuses on data analysis, insights, and reporting."
)


# =========================================================
# 7. UPLOAD DATASET
# =========================================================

if page == "Upload Dataset":
    st.header("1. Upload Dataset")

    st.write("Upload any CSV file to begin your analysis.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_csv(uploaded_file)

        if df is not None:
            st.session_state.df = df
            st.success("Dataset uploaded successfully!")

            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Rows", df.shape[0])

            with col2:
                st.metric("Columns", df.shape[1])

            with col3:
                st.metric("Missing Values", int(df.isnull().sum().sum()))

            with col4:
                st.metric("Duplicate Rows", int(df.duplicated().sum()))

    else:
        st.info("Please upload a CSV file.")


# =========================================================
# 8. DATASET OVERVIEW
# =========================================================

elif page == "Dataset Overview":
    st.header("2. Dataset Overview")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.subheader("Preview")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Basic Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))

        with col4:
            st.metric("Duplicate Rows", int(df.duplicated().sum()))

        st.subheader("Column Details")

        column_details = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(column_details, use_container_width=True)

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(include="all"), use_container_width=True)


# =========================================================
# 9. DATA QUALITY
# =========================================================

elif page == "Data Quality":
    st.header("3. Data Quality Report")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.subheader("Missing Values by Column")
        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum().values,
            "Missing Percentage": ((df.isnull().sum().values / len(df)) * 100).round(2)
        })

        st.dataframe(missing_df, use_container_width=True)

        st.subheader("Duplicate Rows")
        st.write(f"Duplicate rows found: {df.duplicated().sum()}")

        st.subheader("Column Uniqueness")
        unique_df = pd.DataFrame({
            "Column": df.columns,
            "Unique Values": df.nunique().values,
            "Unique Percentage": ((df.nunique().values / len(df)) * 100).round(2)
        })

        st.dataframe(unique_df, use_container_width=True)

        if st.button("Generate AI Data Quality Report"):
            with st.spinner("Generating data quality report..."):
                try:
                    report = generate_data_quality_report(df)
                    st.session_state.data_quality_report = report

                    with open("datasense_data_quality_report.txt", "w", encoding="utf-8") as file:
                        file.write(report)

                    st.success("Data quality report generated!")
                except Exception as e:
                    st.error(f"Failed to generate data quality report: {e}")

        if st.session_state.data_quality_report:
            st.markdown(st.session_state.data_quality_report)

            st.download_button(
                label="Download Data Quality Report",
                data=st.session_state.data_quality_report,
                file_name="datasense_data_quality_report.txt",
                mime="text/plain"
            )


# =========================================================
# 10. VISUALISATIONS
# =========================================================

elif page == "Visualisations":
    st.header("4. Visualisations")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if numeric_columns:
            st.subheader("Numeric Column Distribution")

            selected_num_col = st.selectbox("Choose numeric column", numeric_columns)

            fig, ax = plt.subplots()
            ax.hist(df[selected_num_col].dropna(), bins=30)
            ax.set_title(f"Distribution of {selected_num_col}")
            ax.set_xlabel(selected_num_col)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        if len(numeric_columns) >= 2:
            st.subheader("Scatter Plot")

            x_col = st.selectbox("Choose X-axis", numeric_columns, key="scatter_x")
            y_col = st.selectbox("Choose Y-axis", numeric_columns, key="scatter_y")

            fig2, ax2 = plt.subplots()
            ax2.scatter(df[x_col], df[y_col])
            ax2.set_title(f"{x_col} vs {y_col}")
            ax2.set_xlabel(x_col)
            ax2.set_ylabel(y_col)
            st.pyplot(fig2)

            st.subheader("Correlation Matrix")
            st.dataframe(df[numeric_columns].corr(), use_container_width=True)

        if categorical_columns:
            st.subheader("Category Counts")

            selected_cat_col = st.selectbox("Choose categorical column", categorical_columns)

            top_counts = df[selected_cat_col].value_counts().head(10)

            fig3, ax3 = plt.subplots()
            ax3.bar(top_counts.index.astype(str), top_counts.values)
            ax3.set_title(f"Top Categories in {selected_cat_col}")
            ax3.set_xlabel(selected_cat_col)
            ax3.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig3)

        if not numeric_columns and not categorical_columns:
            st.info("No suitable columns found for visualisation.")


# =========================================================
# 11. AI INSIGHTS
# =========================================================

elif page == "AI Insights":
    st.header("5. AI-Generated Dataset Insights")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        st.write("Generate a professional AI analysis of your uploaded dataset.")

        if st.button("Generate AI Insights"):
            with st.spinner("DataSense AI is analysing your dataset..."):
                try:
                    insights = generate_ai_insights(df)
                    st.session_state.ai_insights = insights

                    with open("datasense_ai_insights.txt", "w", encoding="utf-8") as file:
                        file.write(insights)

                    st.success("AI insights generated!")
                except Exception as e:
                    st.error(f"AI insight generation failed: {e}")

        if st.session_state.ai_insights:
            st.markdown(st.session_state.ai_insights)

            st.download_button(
                label="Download AI Insights",
                data=st.session_state.ai_insights,
                file_name="datasense_ai_insights.txt",
                mime="text/plain"
            )


# =========================================================
# 12. ASK DATASENSE AI
# =========================================================

elif page == "Ask DataSense AI":
    st.header("6. Ask DataSense AI")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        question = st.text_area(
            "Ask a question about your dataset",
            placeholder="Example: What columns have the most missing values? What charts should I create? What patterns can I explore?",
            height=120
        )

        if st.button("Ask"):
            if not question.strip():
                st.error("Please type a question.")
            else:
                with st.spinner("DataSense AI is answering..."):
                    try:
                        answer = answer_dataset_question(df, question)
                        st.session_state.question_answer = answer
                    except Exception as e:
                        st.error(f"Failed to answer question: {e}")

        if st.session_state.question_answer:
            st.markdown(st.session_state.question_answer)


# =========================================================
# 13. DOWNLOAD REPORT
# =========================================================

elif page == "Download Report":
    st.header("7. Download Full Report")

    df = st.session_state.df

    if df is None:
        st.warning("Please upload a dataset first.")
    else:
        full_report = create_full_report(df)

        st.text_area("Report Preview", full_report, height=400)

        st.download_button(
            label="Download Full Analysis Report",
            data=full_report,
            file_name="datasense_full_analysis_report.txt",
            mime="text/plain"
        )


# =========================================================
# 14. PROJECT SUMMARY
# =========================================================

elif page == "Project Summary":
    st.header("8. Project Summary")

    summary_text = """
DataSense AI is an AI-powered CSV data analysis assistant built using Python, Streamlit, pandas, matplotlib, and the OpenAI API.

The app allows users to upload almost any CSV dataset and automatically receive:
- Dataset preview
- Missing value analysis
- Duplicate row check
- Data type summary
- Unique value analysis
- Summary statistics
- Automatic visualisations
- AI-generated insights
- AI data quality report
- Dataset question-answering
- Downloadable analysis report

This project demonstrates practical skills in:
- Data Analytics
- Exploratory Data Analysis
- Data Quality Assessment
- Data Visualisation
- AI-powered insight generation
- Python web app development
- Streamlit UI design
"""

    st.markdown(summary_text)

    st.subheader("LinkedIn Caption Draft")

    linkedin_caption = """
I built DataSense AI, an AI-powered CSV data analysis assistant that allows users to upload datasets and automatically receive exploratory data analysis, missing value checks, duplicate detection, visualisations, AI-generated insights, data quality recommendations, and downloadable reports.

The project combines Python, Streamlit, pandas, matplotlib, and the OpenAI API to automate key stages of the data analysis workflow and make data understanding more accessible.
"""

    st.text_area("Copy this LinkedIn caption:", linkedin_caption, height=220)

    st.download_button(
        label="Download Project Summary",
        data=summary_text + "\n\nLinkedIn Caption:\n" + linkedin_caption,
        file_name="datasense_project_summary.txt",
        mime="text/plain"
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown(
    '<div class="footer">Built with Python, Streamlit, pandas, matplotlib and OpenAI API.</div>',
    unsafe_allow_html=True
)