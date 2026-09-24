from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (ChatPromptTemplate,
                                    FewShotChatMessagePromptTemplate,
                                    MessagesPlaceholder)

from langchain_core.messages import (SystemMessage,HumanMessage,AIMessage)

from langchain_groq import ChatGroq


load_dotenv()

parser = StrOutputParser()

prompt = ChatPromptTemplate.from_template("write a shot poem about {topic}")

llm_model = ChatGroq(model="openai/gpt-oss-120b",
                 temperature=0.5
)

chain = prompt | llm_model | parser

response = chain.invoke({"topic":"Human"})
print(type(response))


#json output parser
from langchain_core.output_parsers import JsonOutputParser
parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_template("Return a JSON object with "
"'name' and age for : {description}")

chain = prompt | llm_model | parser

response = chain.invoke({"description":"I am Akhil Mukesh and I am 29 years old"})
print(type(response))
print(f"Response from JSON Parser: {response}")


#pydantic output parser 
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

class Person(BaseModel):
    name : str = Field(description="This is person name")
    age: int = Field(description="This is going to be persons age")
    occupation : str = Field(description="This is Person occupation")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_template("Return a JSON object with "
"'name' and age and occupation for : {description}"
).partial(format_instruction=parser.get_format_instructions())

chain = prompt | llm_model | parser

response = chain.invoke({"description":"I am Akhil Mukesh and 29 years and software engineer"})

print(response)


#Structred output 

class Movie_Review(BaseModel):
    title: str = Field(description="This is movie title")
    review: str = Field(description="A brief review of the movie")
    rating: int = Field(description= "The rating of the movie out of 10")


#bind the schema to model 
structured_model = llm_model.with_structured_output(Movie_Review)

result = structured_model.invoke("Review: Inceptoin is a mind blowing moview wiht rating 10")
print(result)
