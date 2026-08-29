from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

# ============================================================
# 1. CHECK API KEY
# ============================================================

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found.")

print("Gemini API key found successfully.")


# ============================================================
# 2. EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

print("Embedding model initialized.")


# ============================================================
# 3. CONNECT TO LOCAL QDRANT
# ============================================================

client = QdrantClient(
    path="./qdrant_db"
)

print("Qdrant local database connected.")


# ============================================================
# 4. CONNECT TO EXISTING COLLECTION
# ============================================================

vector_store = QdrantVectorStore(
    client=client,
    collection_name="smartcharge",
    embedding=embeddings
)

print("Vector store connected successfully.")


# ============================================================
# 5. CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ============================================================
# 6. TEST RETRIEVAL
# ============================================================

query = "What factors does SmartCharge AI use to recommend a charging station?"

results = retriever.invoke(query)


print("\n===== RETRIEVAL RESULTS =====")

for i, doc in enumerate(results):
    print(f"\n--- Result {i + 1} ---")
    print(doc.page_content)
    print("Metadata:", doc.metadata)