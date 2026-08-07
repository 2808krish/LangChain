from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert travel guide."),
        ("human", "Tell me about {country}.")
    ]
)

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke(
    {
        "country": "Japan"
    }
)

print(response)