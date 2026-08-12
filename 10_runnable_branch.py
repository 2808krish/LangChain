from langchain_core.runnables import RunnableBranch, RunnableLambda

adult = RunnableLambda(lambda x: f"{x['name']} is an adult.")

minor = RunnableLambda(lambda x: f"{x['name']} is a minor.")

branch = RunnableBranch((lambda x:x["age"]>=18,adult), minor)

response = branch.invoke(
    {
        "name": "Krish",
        "age": 16
    }
)
print(response)