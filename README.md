# DB_Chat_System
Worker Analytics Chatbot Agent

Summary: A local AI-powered analytics agent built using Python, Ollama ("qwen2.5:3b"), and PostgreSQL. It answers natural-language questions about worker data.

Prerequisites

- Python 3.11.0rc2
- PostgreSQL
- Ollama with "qwen2.5:3b"
- Libraries: "psycopg[binary]", "ollama"

Database Details

- Database: "Test_db"
- Schema: "public"
- Main table: "sample_workers"
- Master tables: "social_category_master", "Gender_master"

Questions It Can Answer

- How many workers are there?
- How many workers belong to each gender?
- How many workers belong to each social category?
- Show gender-wise counts for each social category.
- Count workers by occupation or city.
- Calculate average salary or age.
- Filter workers based on available column values.

Operations Supported

- "COUNT" — Count workers or group-wise records.
- "GROUP BY" — Summarize data by gender, social category, occupation, etc.
- "JOIN" — Connect worker data with master tables.
- "AVG", "MIN", "MAX", "SUM" — Perform numerical calculations where applicable.
- "WHERE" — Filter records based on conditions.
- "ORDER BY" — Sort query results.

Setup

1. Install the required Python libraries.
2. Configure PostgreSQL connection details.
3. Ensure the database and required tables exist.
4. Run "ollama pull qwen2.5:3b".
5. Run the Python application.

Command for running the UI- Folder location -m streamlit run app.py
In app.py place that python file in from [Final_agent import (Your python file), describe_data] which you want to run