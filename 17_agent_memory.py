from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

agent = create_agent(
    model=llm,
    tools=[]
)

# Memory for each session
sessions = {}

def chat(session_id, user_message):

    # Create memory for a new session
    if session_id not in sessions:
        sessions[session_id] = []

    # Add user's message
    sessions[session_id].append(
        HumanMessage(content=user_message)
    )

    # Send complete conversation to agent
    result = agent.invoke(
        {
            "messages": sessions[session_id]
        }
    )

    # Get the agent's latest message
    ai_message = result["messages"][-1]

    # Save AI response to memory
    sessions[session_id].append(ai_message)

    return ai_message.content

# Session 1
print("Session 1:")

print(chat(
    "krish_123",
    "Hi, my name is Krish."
))

print(chat(
    "krish_123",
    "What is my name?"
))

# Session 2
print("\nSession 2:")

print(chat(
    "rahul_456",
    "What is my name?"
))