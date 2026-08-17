from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent


load_dotenv()

print("API key loaded:", bool(os.getenv("GOOGLE_API_KEY")))


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


agent = create_agent(
    model=llm,
    tools=[add, multiply]
)

result1 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "My name is Krish."
            }
        ]
    }
)

print("Response 1:")
print(result1["messages"][-1].content)


result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is my name?"
            }
        ]
    }
)

print("Response 2:")
print(result2["messages"][-1].content)