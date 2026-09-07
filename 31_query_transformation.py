import os

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_qdrant import QdrantVectorStore

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from qdrant_client import QdrantClient

from sentence_transformers import CrossEncoder


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. LOAD smartcharge.txt
# ============================================================

loader = TextLoader("smartcharge.txt")

documents = loader.load()

print("Document Loaded:", len(documents))


# ============================================================
# 3. SPLIT INTO CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of Chunks:", len(chunks))


# ============================================================
# 4. CREATE BM25 RETRIEVER
# ============================================================

bm25_retriever = BM25Retriever.from_documents(chunks)

bm25_retriever.k = 2

print("BM25 Retriever initialized")


# ============================================================
# 5. INITIALIZE EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

print("Embedding model initialized")


# ============================================================
# 6. CONNECT TO EXISTING QDRANT
# ============================================================

client = QdrantClient(
    path="./qdrant_data"
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="smartcharge",
    embedding=embeddings
)

print("Qdrant Vector Store connected")


# ============================================================
# 7. CREATE VECTOR RETRIEVER
# ============================================================

vector_retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("Vector Retriever initialized")


# ============================================================
# 8. INITIALIZE GEMINI
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

print("Gemini initialized successfully")


# ============================================================
# 9. QUERY REWRITING PROMPT
# ============================================================

rewrite_prompt = ChatPromptTemplate.from_template(
    """
You are a query optimization system for a SmartCharge AI
EV charging knowledge base.

Rewrite the user's question into a concise search query.

The rewritten query should:
- preserve the user's original intent
- identify the important concepts
- include useful keywords for document retrieval
- remove unnecessary conversational wording
- NOT answer the question

Return ONLY the rewritten search query.

User question:
{question}
"""
)


# ============================================================
# 10. CREATE QUERY REWRITER
# ============================================================

query_rewriter = (
    rewrite_prompt
    | llm
    | StrOutputParser()
)

print("Query Rewriter initialized")


# ============================================================
# 11. RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(results_list, k=60):

    scores = {}
    documents = {}

    for results in results_list:

        for rank, doc in enumerate(results, 1):

            doc_id = doc.page_content

            if doc_id not in scores:
                scores[doc_id] = 0

            scores[doc_id] += 1 / (k + rank)

            documents[doc_id] = doc

    ranked_documents = sorted(
        documents.values(),
        key=lambda doc: scores[doc.page_content],
        reverse=True
    )

    return ranked_documents


# ============================================================
# 12. INITIALIZE RERANKER
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker initialized")


# ============================================================
# 13. USER QUESTION
# ============================================================

question = "I'm almost out of battery. Can I reach a good charger?"


# ============================================================
# 14. QUERY TRANSFORMATION
# ============================================================

rewritten_query = query_rewriter.invoke(
    {
        "question": question
    }
)


# ============================================================
# 15. DISPLAY ORIGINAL + REWRITTEN QUERY
# ============================================================

print("\n========================================")
print("QUERY TRANSFORMATION")
print("========================================")

print("\nOriginal Query:")
print(question)

print("\nRewritten Query:")
print(rewritten_query)


# ============================================================
# 16. BM25 SEARCH USING REWRITTEN QUERY
# ============================================================

bm25_results = bm25_retriever.invoke(
    rewritten_query
)

print("\n========================================")
print("BM25 RESULTS")
print("========================================")

for i, doc in enumerate(bm25_results, 1):

    print(f"\nResult {i}:")
    print(doc.page_content)


# ============================================================
# 17. VECTOR SEARCH USING REWRITTEN QUERY
# ============================================================

vector_results = vector_retriever.invoke(
    rewritten_query
)

print("\n========================================")
print("VECTOR RESULTS")
print("========================================")

for i, doc in enumerate(vector_results, 1):

    print(f"\nResult {i}:")
    print(doc.page_content)


# ============================================================
# 18. HYBRID SEARCH USING RRF
# ============================================================

hybrid_results = reciprocal_rank_fusion(
    [
        bm25_results,
        vector_results
    ]
)


# ============================================================
# 19. DISPLAY HYBRID RESULTS
# ============================================================

print("\n========================================")
print("HYBRID SEARCH RESULTS")
print("========================================")

for i, doc in enumerate(hybrid_results, 1):

    print(f"\nResult {i}:")
    print(doc.page_content)


# ============================================================
# 20. RERANK HYBRID RESULTS
# ============================================================

rerank_pairs = [
    [question, doc.page_content]
    for doc in hybrid_results
]

rerank_scores = reranker.predict(
    rerank_pairs
)


# ============================================================
# 21. SORT DOCUMENTS BY RERANKING SCORE
# ============================================================

reranked_results = sorted(
    zip(hybrid_results, rerank_scores),
    key=lambda x: x[1],
    reverse=True
)


# ============================================================
# 22. DISPLAY RERANKED RESULTS
# ============================================================

print("\n========================================")
print("RERANKED RESULTS")
print("========================================")

for i, (doc, score) in enumerate(reranked_results, 1):

    print(f"\nResult {i} | Score: {score:.4f}")
    print(doc.page_content)


# ============================================================
# 23. SELECT TOP DOCUMENTS
# ============================================================

top_results = reranked_results[:3]


# ============================================================
# 24. BUILD CONTEXT
# ============================================================

context = "\n\n".join(
    doc.page_content
    for doc, score in top_results
)


# ============================================================
# 25. FINAL ANSWER PROMPT
# ============================================================

answer_prompt = ChatPromptTemplate.from_template(
    """
You are SmartCharge AI, an intelligent EV charging assistant.

Answer the user's question using ONLY the provided context.

If the context does not contain enough information,
say that the information is not available.

Do not invent facts.

User Question:
{question}

Context:
{context}

Answer:
"""
)


# ============================================================
# 26. FINAL ANSWER CHAIN
# ============================================================

answer_chain = (
    answer_prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 27. GENERATE FINAL ANSWER
# ============================================================

answer = answer_chain.invoke(
    {
        "question": question,
        "context": context
    }
)


# ============================================================
# 28. DISPLAY FINAL ANSWER
# ============================================================

print("\n========================================")
print("SMARTCHARGE AI ANSWER")
print("========================================")

print(answer)