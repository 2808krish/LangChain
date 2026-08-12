from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

python_prompt = ChatPromptTemplate.from_template(
    "You are a Python expert. Answer this question clearly:\n{question}"
)

general_prompt = ChatPromptTemplate.from_template(
    "You are a general assistant. Answer this question clearly:\n{question}"
)

python_chain = python_prompt | llm | StrOutputParser()
general_chain = general_prompt | llm |StrOutputParser()

branch = RunnableBranch(
    (
        lambda x: "python" in x["question"].lower(), python_chain
    ),
    general_chain
)

response = branch.invoke(
    {
        "question" : "What is a list comprehension in Python?"
    }
)

print(response)