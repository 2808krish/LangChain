from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda

load_dotenv()

class langChain(BaseModel):
    country : str
    capital : str
    language : str
    
parser = PydanticOutputParser(pydantic_object=langChain)

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

template = PromptTemplate(
    template="""
Answer the following question.

{format_instructions}

Country: {country}
Language: {language}
""",
    input_variables = ["country", "language"],
    partial_variables =
    {
        "format_instructions": parser.get_format_instructions()
    }
)

capital_chain = template | llm | parser

def extract_country(data):
    return data["country"]

main_chain = RunnableParallel(
    country=RunnableLambda(extract_country),
    capital=capital_chain
)

response = main_chain.invoke(
    {
        "country":"Spain",
        "language":"Spanish"
    }
)

print(response)