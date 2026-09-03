from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from qdrant_client import QdrantClient


# 1. LOAD smartcharge.txt

loader = TextLoader("smartcharge.txt")

documents = loader.load()

print("Document Loaded:", len(documents))


# 2. SPLIT INTO CHUNKS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of Chunks:", len(chunks))


# 3. CREATE BM25 RETRIEVER

bm25_retriever = BM25Retriever.from_documents(chunks)

bm25_retriever.k = 2

print("BM25 Retriever initialized")


# 4. INITIALIZE EMBEDDINGS

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

print("Embedding model initialized")


# 5. CONNECT TO EXISTING QDRANT

client = QdrantClient(
    path="./qdrant_data"
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="smartcharge",
    embedding=embeddings
)

print("Qdrant Vector Store connected")


# 6. CREATE VECTOR RETRIEVER

vector_retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

print("Vector Retriever initialized")

# 7. TEST QUERY

query = "What is SmartCharge AI?"


# 8. BM25 SEARCH

bm25_results = bm25_retriever.invoke(query)

print("\n===== BM25 RESULTS =====")

for i, doc in enumerate(bm25_results, 1):
    print(f"\nResult {i}:")
    print(doc.page_content)


# 9. VECTOR SEARCH

vector_results = vector_retriever.invoke(query)

print("\n===== VECTOR RESULTS =====")

for i, doc in enumerate(vector_results, 1):
    print(f"\nResult {i}:")
    print(doc.page_content)
    
    
#10 Reciprocal Rank Fusion (RRF)

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

# 13. COMBINE BM25 + VECTOR RESULTS


hybrid_results = reciprocal_rank_fusion(
    [
        bm25_results,
        vector_results
    ]
)

# 14. DISPLAY HYBRID SEARCH RESULTS

print("\n========================================")
print("HYBRID SEARCH RESULTS")
print("========================================")

for i, doc in enumerate(hybrid_results, 1):

    print(f"\nResult {i}:")
    print(doc.page_content)