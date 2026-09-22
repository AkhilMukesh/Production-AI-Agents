from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
prompt = ChatPromptTemplate.from_template("""
Create a market tagline for product name '{product}' targeting '{audience}'.

""")

model = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature=0.6
)

parser = StrOutputParser()
chain = prompt | model | parser

response = chain.invoke({"product" :"Shampoo","audience":"people"})

print(f"The marketing ad goes like this:{response}")