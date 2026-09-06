import os
from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# 1. LOAD ENVIRONMENT VARIABLES

load_dotenv()

# 2. CHECK GEMINI API KEY

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY or GOOGLE_API_KEY not found."
    )

print("Gemini API key found successfully.")


# 3. INITIALIZE GEMINI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

print("Gemini initialized successfully.")

# 4. QUERY REWRITING PROMPT

prompt = ChatPromptTemplate.from_template(
    """
You are a query optimization system for a SmartCharge AI
EV charging knowledge base.

Rewrite the user's question into a concise search query
containing the important concepts and keywords needed for
document retrieval.

Do not answer the question.
Return only the rewritten search query.

User question:
{question}
"""
)


# 5. CREATE QUERY TRANSFORMATION CHAIN

query_rewriter = (prompt | llm | StrOutputParser())

# 6. CONNECT TO EXISTING QDRANT COLLECTION

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

print("Embedding model initialized successfully.")

client = QdrantClient(
    path="./qdrant_data"
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="smartcharge",
    embedding=embeddings
)

print("Connected to Qdrant successfully.")

# 7. CREATE RETRIEVER

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5
    }
)
print("Retriever initialized successfully.")

# 8. TEST QUERY

question = "I'm almost out of battery. Can I reach a good charger?"

# 9. TRANSFORM QUERY

rewritten_query = query_rewriter.invoke(
    {
        "question": question
    }
)

# 10. DISPLAY ORIGINAL + REWRITTEN QUERY

print("\n" + "=" * 70)
print("QUERY TRANSFORMATION")
print("=" * 70)

print("\nOriginal Query:")
print(question)

print("\nRewritten Query:")
print(rewritten_query)

# 11. RETRIEVE USING ORIGINAL QUERY

original_results = retriever.invoke(question)

print("\n" + "=" * 70)
print("RETRIEVAL USING ORIGINAL QUERY")
print("=" * 70)

for rank, document in enumerate(original_results, 1):

    print(f"\nRank {rank}")
    print(f"Content: {document.page_content}")


# 12. RETRIEVE USING REWRITTEN QUERY

rewritten_results = retriever.invoke(
    rewritten_query
)

print("\n" + "=" * 70)
print("RETRIEVAL USING REWRITTEN QUERY")
print("=" * 70)

for rank, document in enumerate(rewritten_results, 1):

    print(f"\nRank {rank}")
    print(f"Content: {document.page_content}")