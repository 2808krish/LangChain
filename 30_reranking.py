import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

# 1 LOAD ENVIRONMENT VARIABLES

load_dotenv()


# 2. CHECK GEMINI API KEY

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY or GOOGLE_API_KEY not found."
    )

print("Gemini API key found successfully.")


# 3. INITIALIZE EMBEDDING MODEL

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

print("Embedding model initialized successfully.")


# 4. CONNECT TO EXISTING QDRANT COLLECTION

client = QdrantClient(
    path="./qdrant_data"
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="smartcharge",
    embedding=embeddings
)

print("Connected to Qdrant successfully.")

# 5. CREATE RETRIEVER

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5
    }
)

print("Retriever initialized successfully.")


# 6. INITIALIZE CROSS-ENCODER RERANKER

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

print("Reranker initialized successfully.")


# 7. USER QUERY

query = input(
    "\nEnter your question: "
)

# 8. RETRIEVE CANDIDATE DOCUMENTS

retrieved_documents = retriever.invoke(query)

print("\n" + "=" * 70)
print("INITIAL RETRIEVAL RESULTS")
print("=" * 70)

for rank, document in enumerate(retrieved_documents, 1):
    print(f"\nRank {rank}")
    print(f"Content: {document.page_content}")

# 9. PREPARE QUERY + DOCUMENT PAIRS

pairs = [
    [query, document.page_content]
    for document in retrieved_documents
]

# 10. GENERATE RERANKING SCORES

scores = reranker.predict(pairs)

# 11. COMBINE DOCUMENTS WITH SCORES

reranked_results = list(zip(retrieved_documents, scores))

# 12. SORT BY RELEVANCE

reranked_results.sort(
    key=lambda x: x[1],
    reverse=True
)

# 13. DISPLAY RERANKED RESULTS

print("\n" + "=" * 70)
print("RERANKED RESULTS")
print("=" * 70)

for rank, (document,score) in enumerate(reranked_results,1):

    print(f"\nRank {rank}")
    print(f"Relevance Score: {score:.4f}")
    print(f"Content: {document.page_content}")

# 14. SELECT TOP-N DOCUMENTS

top_n = 3

final_documents = reranked_results[:top_n]

print("\n" + "=" * 70)
print(f"FINAL TOP {top_n} DOCUMENTS")
print("=" * 70)

for rank, (document, score) in enumerate(final_documents,1):

    print(f"\nRank {rank}")
    print(f"Score: {score:.4f}")
    print(f"Content: {document.page_content}")

# 15. BUILD FINAL CONTEXT

context = "\n\n".join(document.page_content for document, score in final_documents)

print("\n" + "=" * 70)
print("FINAL CONTEXT SENT TO GEMINI")
print("=" * 70)

print(context)

# 16. CREATE GEMINI MODEL

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

print("\nGemini initialized successfully.")

# 17. CREATE RAG PROMPT

prompt = ChatPromptTemplate.from_template(
    """
You are an intelligent assistant for SmartCharge AI.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that the available information is insufficient.

Do not invent information.

Context:
{context}

User Question:
{question}

Answer:
"""
)

# 18. CREATE RAG CHAIN

chain = (prompt | llm | StrOutputParser())

# 19. GENERATE FINAL ANSWER

answer = chain.invoke(
    {
        "context": context,
        "question": query
    }
)

# 20. DISPLAY FINAL ANSWER

print("\n" + "=" * 70)
print("SMARTCHARGE AI FINAL ANSWER")
print("=" * 70)

print("\n" + answer)

# 21. PIPELINE SUMMARY

print("\n" + "=" * 70)
print("COMPLETE RERANKING RAG PIPELINE")
print("=" * 70)

print(
    """
User Question
      ↓
Qdrant Retriever
      ↓
Top 5 Candidate Documents
      ↓
Cross-Encoder Reranker
      ↓
Relevance Scores
      ↓
Sort by Relevance
      ↓
Top 3 Documents
      ↓
Final Context
      ↓
Gemini
      ↓
Final Answer
"""
)