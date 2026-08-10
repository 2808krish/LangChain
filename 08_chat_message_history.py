from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
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

history = InMemoryChatMessageHistory()

history.add_user_message("Hi, my name is Krish.")
history.add_ai_message("Nice to meet you, Krish!")

response = chain.invoke(
    {
        "history": history.messages,
        "question": "What is my name?"
    }
)

print(response)