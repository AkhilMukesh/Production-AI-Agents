from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage,SystemMessage

load_dotenv()


#chat prompt template

prompt = ChatPromptTemplate.from_template("Tell me {adjective} joke about {topic}. ")
messages = prompt.format_messages(adjective='Funny', topic='Dog')

print(messages)


#multi message template 
multi_prompt = ChatPromptTemplate.from_messages(
[
    ("system", "You are a helpful assistant taht translats {input_language} tp {output_language}. "), 
    ("human", "Transalate the following text:{text}"),
]
)

multi_message = multi_prompt.format_messages(input_language = "English",
                                             output_language = "French",
                                             text = "I love AI")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
)

response = model.invoke(multi_message);

print(response.content)


#message types
from langchain_core.messages import (AIMessage,HumanMessage,SystemMessage,ToolMessage,ChatMessage)

messages = [HumanMessage(content="Hello"),
            AIMessage(content="Hi There! how cam i assist you"),
            SystemMessage(content="This is System message"),
           # ToolMessage(content="This is tool message"),
            #ChatMessage(content="This is chat message")
            ]

#feew shots example 
from langchain_core.prompts import FewShotChatMessagePromptTemplate

example = [{"input":"happy", "output":"sad"},
           {"input":"tall", "output":"short"}]

example_prompt = ChatPromptTemplate.from_messages([
    ("human","{input}"),
    ("ai","{output}")

])

fewshot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt,
                                                  examples=example)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","Give the opposite of the words"),
        fewshot_prompt,
        ("human","{input}")
    ]
)

response_few_shot = model.invoke(final_prompt.format_messages(input="happy"))
print(response_few_shot.content)