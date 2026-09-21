"""Lesson 1"""
from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version
from importlib.metadata import version
from langchain_groq import ChatGroq
import streamlit as st


core_version = version("langchain-core")
lg_version = version("langgraph")


# UI
st.title("🤖 Production AI Agents")

st.write("LangChain + Groq")

st.write(f"LangChain Core Version: {core_version}")
st.write(f"LangGraph Version: {lg_version}")


def main():

    print("Hello from langchain")

    llm_groq = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )

    user_input = st.text_input("Enter your question:")

    if user_input:

        response_groq = llm_groq.invoke(user_input)

        st.write("### Groq Response")
        st.write(response_groq.content)

        print(f"Response from ChatGroq: {response_groq}")

    print("Setup complete")


if __name__ == "__main__":
    main()

