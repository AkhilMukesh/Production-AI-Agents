
from dotenv import load_dotenv
from importlib.metadata import version

import streamlit as st
from langchain_groq import ChatGroq


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()


# --------------------------------------------------
# Get package versions
# --------------------------------------------------
core_version = version("langchain-core")
lg_version = version("langgraph")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Production AI Agent",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("🤖 Production AI Agent")
st.caption("LangChain + LangGraph + Groq")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.header("⚙️ Configuration")

    st.write("**LLM Provider:** Groq")
    st.write("**Model:** openai/gpt-oss-120b")

    st.divider()


    st.write(f"LangChain Core: `{core_version}`")
    st.write(f"LangGraph: `{lg_version}`")


# --------------------------------------------------
# Create Groq LLM
# --------------------------------------------------
@st.cache_resource
def get_llm():

    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )


llm_groq = get_llm()


# --------------------------------------------------
# Chat history
# --------------------------------------------------
if "messages" not in st.session_state:

    st.session_state.messages = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------
user_input = st.chat_input(
    "Ask something..."
)


# --------------------------------------------------
# Process user message
# --------------------------------------------------
if user_input:

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)


    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = llm_groq.invoke(user_input)

            answer = response.content

            st.markdown(answer)


    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
