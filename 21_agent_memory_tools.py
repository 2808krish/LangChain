from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """Add two Numbers."""
    return a + b

llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

agent = create_agent(
    model= llm,
    tools =[add]
)

sessions = {}

def chat(session_id, user_message):
    if session_id not in sessions:
        sessions[session_id] = []
        
    sessions[session_id].append({
        "role" : "user",
        "content" : user_message
    })
    
    response = agent.invoke({
        "messages": sessions[session_id]
    })
    
    sessions[session_id].extend(response["messages"][len(sessions[session_id]):])
    
    return response["messages"][-1].content


print("Session 1:\n")
print(chat(
    "krish_123",
    "My name is Krish"
))

print(chat(
    "krish_123",
    "What is my name?"
))

print(chat(
    "krish_123",
    "What is 25 + 17?"
))

print("Session 2:\n")
print(chat(
    "rahul_456",
    "What is my name?"
))