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

@tool
def multiply(a:int, b:int) -> int:
    """Multiply two numbers."""
    return a*b

tools = {
    "add": add,
    "multiply": multiply
}

# 2. Create the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)


# 3. Give the tool to the LLM
llm_with_tools = llm.bind_tools([add, multiply])


# 4. User's question
user_message = HumanMessage(
    content= "Calculate 10 + 5 and then multiply the result by 3."
)

response = llm_with_tools.invoke([user_message])

messages = [
    user_message,
    response
]

while response.tool_calls:

    for tool_call in response.tool_calls:

        selected_tool = tools[tool_call["name"]]

        result = selected_tool.invoke(tool_call["args"])

        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        )

        messages.append(tool_message)

    response = llm_with_tools.invoke(messages)
    messages.append(response)

print(response.text)