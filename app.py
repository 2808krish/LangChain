from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}")
    ]
)

parser = StrOutputParser()

chain = prompt | llm | parser

history = [
    HumanMessage(content="Hi"),
    AIMessage(content="Hello!"),
    HumanMessage(content="My name is Walter Hartwell White"),
    AIMessage(content="Nice to meet you!")
]

response = chain.invoke(
    {
        "history": history,
        "question": "What is my name?"
    }
)

print(response)