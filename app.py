from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash"
)

template = PromptTemplate.from_template(
    "What is the capital of {country}?, Answer in {language}"
)
prompt = template.format(country="Japan", language= "Hindi")

response= llm.invoke(prompt)

print(response.content[0]["text"])