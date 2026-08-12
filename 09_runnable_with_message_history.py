from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
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
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
        
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key= "question",
    history_messages_key="history"
)

response = chain_with_history.invoke(
    {
        "question" : "Hi, my name is Krish."
    },
    config={
        "configurable":{
            "session_id":"krish_123"
        }
    }
)

print(response)

response2 = chain_with_history.invoke(
    {
        "question" : "What is my Name?"
    },
    config={
        "configurable":{
            "session_id": "rahul_456"
        }
    }
)
print(response2)