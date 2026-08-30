import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# ============================================================
# 1. CHECK API KEY
# ============================================================

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found")

print("Gemini API key found successfully.")


# ============================================================
# 2. INITIALIZE EMBEDDING MODEL
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

print("Embedding model initialized.")


# ============================================================
# 3. CONNECT TO LOCAL QDRANT
# ============================================================

client = QdrantClient(
    path="./qdrant_data"
)

print("Qdrant local database connected.")

print("\n===== AVAILABLE COLLECTIONS =====")

collections = client.get_collections()

for collection in collections.collections:
    print(collection.name)


# ============================================================
# 4. CONNECT TO EXISTING COLLECTION
# ============================================================

vector_store = QdrantVectorStore(
    client=client,
    embedding=embeddings,
    collection_name="smartcharge"
)

print("Vector store connected successfully.")


# ============================================================
# 5. CREATE MMR RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10
    }
)


# ============================================================
# 6. ASK QUESTION
# ============================================================

query = "What factors does SmartCharge AI use for recommendations?"


# ============================================================
# 7. RETRIEVE USING MMR
# ============================================================

results = retriever.invoke(query)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n===== MMR RETRIEVED RESULTS =====")

for i, doc in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)