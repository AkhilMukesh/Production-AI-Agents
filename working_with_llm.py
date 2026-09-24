"""
Working with LLM's in Langchain v.1
multiple providers, cost optimization, configuration and streaming

"""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()


def demo_messages():


    model = ChatGroq(model="openai/gpt-oss-120b",
                 temperature=0.5,
                 streaming = True,
                 max_retries=2)
    # response = model.invoke("What is India's capital. answer in one word");
    # print(f"Repsonse is : {response.content}")

    #using message objects (more control over roles)
    messages = [
        SystemMessage(content = "You are helpful assitant"),
        HumanMessage(content="who is the PM of India. Answer in one line.")

    ]

    response = model.invoke(messages)
    print(f"Reponse from LLM AI: {response.content}")

    #multi turn conversion using message objects 
    messages.append(response)
    messages.append(HumanMessage(content="what about president? "))

    print("Multi conversiotn chat: ")
    response = model.invoke(messages)
    print(response.content)





if __name__ == "__main__":
    demo_messages()






