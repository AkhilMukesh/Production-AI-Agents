from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough, RunnableLambda

load_dotenv()

model = ChatGroq(
                model = "openai/gpt-oss-120b",
                temperature=0.6
                )

def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template("Summarize the following text in one word: {text}")

    parser = StrOutputParser()

    chain = prompt | model | parser
    response = chain.invoke({'text': "Hello what a beutiful day and welcome to langchain framework"})
    print(f"Response from basic chain :{response}")

#run multiple chain in parallel
def demo_parallel_chain():
    summarize_prompt = ChatPromptTemplate.from_template("summarize: {text}")
    keywords_prompt = ChatPromptTemplate.from_template("Extract the keywords : {text}")
    sentiment_prompt = ChatPromptTemplate.from_template("sentiment : {text}")


    parser = StrOutputParser()

    #parallele exeuction
    chain = RunnableParallel(summary = summarize_prompt | model | parser, 
                             keyowrds = keywords_prompt | model | parser,
                            sentiment =  sentiment_prompt | model | parser
                             )

    text = """The new AI feature are absolutley incredible. Useres are loving the faster response times and impoved accuracy.
    Overall the product lauch has been masive success with record-breaking adoption rates"""

    result = chain.invoke({"text":text})
    print("parallel Analysis result")
    print(f"summary:{result['summary']}")
    print(f"Keywords:{result['keyowrds']}")
    print(f"sentiment:{result['sentiment']}")

def demo_pass_through():
    prompt = ChatPromptTemplate.from_template(
        "original question: {question}"
        "context:{context}"
        "Anser the question based on context"
    )

    def fake_retriver(input_dict):
        return "Langchian was delveloped by harrison chase in 2022"

    chain = (
        RunnableParallel(
        context = RunnableLambda(fake_retriver),
        question = RunnablePassthrough()
        ) | RunnableLambda(
            lambda x: {"context":x["context"] ,
                        "question":x["question"]["question"]} 
                        ) | prompt | model| StrOutputParser()
    )
    result = chain.invoke({"question":"who created langchain?"})
    print(result)




if __name__ == "__main__":
   # demo_basic_chain()
   # demo_parallel_chain()
    demo_pass_through()

