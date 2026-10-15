import streamlit as st
from Final_agent import ask_agent, describe_data

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Worker Analytics AI",
    page_icon="📊",
    layout="wide",
)

# -----------------------------
# Session state
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# Header
# -----------------------------

st.title("📊 Worker Analytics AI")
st.caption("Ask questions about your PostgreSQL worker database.")

st.divider()

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:
    st.header("⚙️ Dashboard")

    st.write("**AI Model:** `qwen2.5:3b`")
    st.write("**Database:** `Test_db`")
    st.write("**Table:** `public.sample_workers`")

    st.divider()

    if st.button("🔌 Check Database Connection", use_container_width=True):
        try:
            columns = describe_data()
            st.success("Database connected successfully!")

            with st.expander("Table columns"):
                for column in columns:
                    st.write(
                        f"- {column['column_name']} "
                        f"({column['data_type']})"
                    )

        except Exception as exc:
            st.error(f"Connection failed: {exc}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Powered by Streamlit, Ollama and PostgreSQL.")

# -----------------------------
# Suggested questions
# -----------------------------

st.subheader("Try asking")

suggestions = [
    "How many workers are registered?",
    "What is the maximum salary in Mumbai?",
    "Show worker count by gender",
    "Show occupation-wise worker count",
]

cols = st.columns(2)

for i, question in enumerate(suggestions):
    with cols[i % 2]:
        if st.button(question, key=f"suggestion_{i}",
                     use_container_width=True):
            st.session_state.pending_question = question

# -----------------------------
# Display chat history
# -----------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# Chat input
# -----------------------------

question = st.chat_input("Ask a question about your worker data...")

if question:
    st.session_state.pending_question = question

if "pending_question" in st.session_state:
    current_question = st.session_state.pop("pending_question")

    st.session_state.messages.append({
        "role": "user",
        "content": current_question,
    })

    with st.chat_message("user"):
        st.markdown(current_question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            try:
                answer = ask_agent(current_question)
            except Exception as exc:
                answer = f"Error: {exc}"

        if "SQL Query:" in answer:
            result_text, sql_section = answer.split(
                "SQL Query:", 1
            )

            st.markdown(result_text.strip())

            sql_text = sql_section
            params_text = None

            if "Parameters:" in sql_section:
                sql_text, params_text = sql_section.split(
                    "Parameters:", 1
                )

            with st.expander("🧾 SQL Query", expanded=True):
                st.code(sql_text.strip(), language="sql")

                if params_text is not None:
                    st.write("**Query Parameters**")
                    st.code(params_text.strip(), language="python")
        else:
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
    })