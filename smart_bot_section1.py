import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from langsmith import traceable, Client
import os


load_dotenv()


# ============================================================
# LangSmith configuration
# ============================================================

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_PROJECT", "Smart Q&A Bot")

    print(
        f"LangSmith is configured. "
        f"PROJECT: {os.getenv('LANGSMITH_PROJECT')}"
    )
else:
    print("LangSmith API key not found")


# ============================================================
# Schema Definition
# ============================================================

class QAResponse(BaseModel):

    answer: str = Field(
        description="The answer to the user question"
    )

    confidence: str = Field(
        description="Confidence level: high, medium, low"
    )

    reasoning: str = Field(
        description="The reasoning behind the answer"
    )

    follow_up_questions: List[str] = Field(
        description="A list of follow up questions related to topic",
        default_factory=list
    )

    sources_needed: bool = Field(
        description="Indicates whether sources are needed",
        default=False
    )


# ============================================================
# Smart Bot
# ============================================================

class SmartBot:

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temprature: float = 0.3
    ):

        self.model = ChatGroq(
            model=model_name,
            temperature=temprature
        ).with_structured_output(QAResponse)

        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a Knowledgeable Assistant.

Your guidelines:

- Answer questions accurately and concisely.
- Be honest about uncertainty - set confidence to 'low' if unsure.
- Provide clear reasoning for your answer.
- Suggest relevant follow-up questions.
- Indicate if external sources would help.

Always respond with accurate helpful information.
"""
            ),
            (
                "human",
                "{question}"
            )
        ])

        self.chain = self.prompt | self.model


    @traceable(
        name="ask_question",
        run_type="chain"
    )
    def ask(
        self,
        question: str
    ) -> QAResponse:

        try:

            response = self.chain.invoke(
                {"question": question}
            )

            return response

        except Exception as e:

            return QAResponse(
                answer="I am sorry, I could not process your question at this time.",
                confidence="low",
                reasoning=str(e),
                follow_up_questions=["Try again later"],
                sources_needed=True
            )


    @traceable(
        name="ask_batch",
        run_type="chain"
    )
    def ask_batch(
        self,
        question: List[str]
    ) -> List[QAResponse]:

        """Ask multiple questions in parallel"""

        inputs = [
            {"question": q}
            for q in question
        ]

        return self.chain.batch(inputs)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Smart Q&A Bot",
    page_icon="🤖",
    layout="centered"
)


# ============================================================
# Header
# ============================================================

st.title("🤖 Smart Q&A Bot")

st.write(
    "Ask a question and get an AI-generated answer "
    "with confidence, reasoning, and follow-up questions."
)


# ============================================================
# Create Bot
# ============================================================

@st.cache_resource
def create_bot():

    return SmartBot()


bot = create_bot()


# ============================================================
# Question Input
# ============================================================

question = st.text_area(
    "Ask your question",
    placeholder="Example: Explain the theory of relativity...",
    height=120
)


# ============================================================
# Ask Button
# ============================================================

if st.button(
    "Ask Question",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            response = bot.ask(question)


        # ====================================================
        # Answer
        # ====================================================

        st.subheader("💡 Answer")

        st.write(response.answer)


        # ====================================================
        # Confidence
        # ====================================================

        st.subheader("🎯 Confidence")

        st.write(response.confidence)


        # ====================================================
        # Reasoning
        # ====================================================

        with st.expander("🧠 Reasoning"):

            st.write(response.reasoning)


        # ====================================================
        # Follow-up Questions
        # ====================================================

        st.subheader("🔎 Follow-up Questions")

        if response.follow_up_questions:

            for follow_up in response.follow_up_questions:

                st.write(f"• {follow_up}")

        else:

            st.write("No follow-up questions.")


        # ====================================================
        # Sources
        # ====================================================

        st.subheader("📚 Sources Needed")

        if response.sources_needed:

            st.warning(
                "External sources may be useful for this question."
            )

        else:

            st.success(
                "External sources are not necessary."
            )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "Powered by LangChain + Groq + LangSmith"
)