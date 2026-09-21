"""
Lesson 2
langchian core concepts - LCEL and Runnables
"""
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

def demo_basic_chain():
    """
    Demo basic chain using LCEL and Runnables.
    """

    #Component 1 : Define the prompt tempate using LCEL
    prompt = ChatPromptTemplate.from_template("You are helpful assistant. " \
                                    "Answer in one sentence: {question}")
    model = ChatGroq(
        model = "openai/gpt-oss-120b",
        temperature=0.6
    )

    parser = StrOutputParser()

    #compose with pipe operator 
    chain = prompt | model | parser

    #execute the chian with input 
    result = chain.invoke({"question": "what is langchain? "})
    print(f"Response : {result}")

    return chain


def demo_batch_execution():
    """Demo batch execution for multiple inputs"""

    prompt = ChatPromptTemplate.from_template("Translate to hindi: {text}")
    model = ChatGroq(
            model = "openai/gpt-oss-120b",
            temperature=0.6
        )
    parser = StrOutputParser()
    chain = prompt | model | parser
    #batch - run with multiple inputs 
    inputs= [
        {"text":"Hello"},
        {"text":"what is your name? "},
        {"text":"where is AI? "}
        ]
    result = chain.batch(inputs)

    for text in zip(inputs,result):
        print(f"Input:{text[0]['text']} => Output:{text[1]}")

def demo_streaming():
    """Demonstarte the streaming for real-time input"""
    prompt = ChatPromptTemplate.from_template("write a haiku about: {topic}")
    model = model = ChatGroq(
                model = "openai/gpt-oss-120b",
                temperature=0.6
            )
    parser = StrOutputParser()

    chain = prompt | model | parser

    #streaming - run with strarming enabled 
    print("Streaming output :") 
    for chunk in chain.stream({"topic":"nature"}):
        print(chunk, end="", flush=True)
    print("")

def demo_schema_inspectoin():
    """Demonstarte input/output schema inspection"""
    prompt = ChatPromptTemplate.from_template("write a haiku : {text}")
    model = model = ChatGroq(
                model = "openai/gpt-oss-120b",
                temperature=0.6
                )
    parser = StrOutputParser()
    
    chain = prompt | model | parser

    #inspect inout and output schemas 
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"input schema: {input_schema}")
    print(f"output schema: {output_schema}")
if __name__ == "__main__":
    #demo_basic_chain()
    #demo_batch_execution()
    #demo_streaming()
    demo_schema_inspectoin()