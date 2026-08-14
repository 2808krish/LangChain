from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a*b

llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

agent = create_agent(
    model= llm,
    tools= [add, multiply]
)

result = agent.invoke(
    {
        "messages":[
            {
                "role" : "user", 
                "content" : "What is 10 + 5, and then multiply the result by 3?"
            }
        ]
    }
)

print(result)

