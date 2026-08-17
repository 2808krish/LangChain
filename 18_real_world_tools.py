from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool

load_dotenv()

@tool
def get_weather(city: str) -> str:
    """Get the weather information for a city."""
    
    weather_data = {
        "Ahmedabad": "32°C and sunny",
        "Mumbai": "29°C and cloudy",
        "Delhi": "35°C and clear"
    }
    
    return weather_data.get(city, "Weather information is not avaliable.")

llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

agent = create_agent(
    model = llm,
    tools=[get_weather]
)

response = agent.invoke(
    {
        "messages" :[
            {
                "role" : "user",
                "content" : "What's the weather in Ahmedabad."
            }
        ]
    }
)

print(response["messages"][-1].content)