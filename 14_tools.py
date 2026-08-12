from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage


load_dotenv()


# 1. Create the tool
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


# 2. Create the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# 3. Give the tool to the LLM
llm_with_tools = llm.bind_tools([add])


# 4. User's question
user_message = HumanMessage(
    content="What is 25 + 17?"
)


# 5. Ask the LLM
response = llm_with_tools.invoke(
    [user_message]
)

print("LLM tool call:")
print(response.tool_calls)


# 6. Get the tool call
tool_call = response.tool_calls[0]


# 7 Execute the actual Python tool
result = add.invoke(tool_call["args"])

print("Tool result:")
print(result)


# 8 the tool result back to the LLM
tool_message = ToolMessage(
    content=str(result),
    tool_call_id=tool_call["id"]
)


# 9 the final human-readable answer
final_response = llm_with_tools.invoke(
    [
        user_message,
        response,
        tool_message
    ]
)


# 10 final answer
print("Final answer:")
print(final_response.content)