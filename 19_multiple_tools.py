from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()


# 1. Create multiple tools

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def get_weather(city: str) -> str:
    """Get the weather information for a city."""

    weather_data = {
        "Ahmedabad": "32°C and sunny",
        "Mumbai": "29°C and cloudy",
        "Delhi": "35°C and clear"
    }

    return weather_data.get(
        city,
        "Weather information is not available."
    )


# 2. Create the LLM

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# 3. Give all tools to the agent

agent = create_agent(
    model=llm,
    tools=[add, multiply, get_weather]
)


# 4. Test different questions

questions = [
    "What is 15 + 7?",
    "What is 8 multiplied by 6?",
    "What's the weather in Ahmedabad?"
]


# 5. Ask the agent each question

for question in questions:

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    print("\nQuestion:", question)
    print("Answer:", response["messages"][-1].content)