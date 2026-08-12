from langchain_core.runnables import RunnableLambda, RunnableSequence
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

step1 = RunnableLambda(lambda x:x["name"])

step2 = RunnableLambda(lambda x:x.upper())

step3 = RunnableLambda(lambda x: f"Hello, {x}!")

chain = RunnableSequence(
    step1,
    step2,
    step3
)

response = chain.invoke(
    {
        "name" : "Krish"
    }
)

print(response)