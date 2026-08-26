from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


# 1. Load document

loader = TextLoader("smartcharge.txt")
documents = loader.load()

print("Document Loaded:", len(documents))


# 2. Split document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of Chunks:", len(chunks))


# 3. Gemini Embeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

print("Embedding model initialized")


# 4. Create Qdrant Vector Store

vector_store = QdrantVectorStore.from_documents(
    documents= chunks,
    embedding= embeddings,
    location= ":memory:",
    collection_name= "smartcharge"
)
print("Qdrant Vector Store created successfully")


# 5. Test retrieval

results = vector_store.similarity_search(
    "What is SmartCharge AI?",
    k=2
)

print("\n===== RETRIEVED DOCUMENTS =====")

for i, doc in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(doc.page_content)