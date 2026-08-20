from pydantic import BaseModel
import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

class Student(BaseModel):
    name: str
    age: int
    skills: list[str]    
    
llm = ChatGoogleGenerativeAI(model= "gemini-3.6-flash")

structured_llm = llm.with_structured_output(Student)

response = structured_llm.invoke(
    "My name is Krish, I am 21 years old and I know Python, FastAPI and LangChain."
)

print(response)
print(response.name)
print(response.age)
print(response.skills)