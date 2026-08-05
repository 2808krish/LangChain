from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel

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

chain = template | llm | parser

response = chain.invoke(
    {
        "country":"Spain",
        "language":"Spanish"
    }
)

print(response)