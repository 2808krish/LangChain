from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from qdrant_client import QdrantClient

# 1 Load environment variables
load_dotenv()

# 2 Load smartcharge.txt
loader = TextLoader("smartcharge.txt")
documents = loader.load()
print("Documents Loaded:", len(documents))

# 3 Create Parent Splitter
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
parent_documents = parent_splitter.split_documents(documents)
print("Number of Parent Documents:", len(parent_documents))

# 4 Create Child Splitter
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=50
)

print("Parent and Child Splitters initialized")


# 5 Initialize Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
print("Embedding model initialized")

# 6 Connect to local Qdrant
client = QdrantClient(path="./qdrant_data")
print("Qdrant client initialized")

# 7 Create Parent/Child Vector Store
vector_store = QdrantVectorStore(
    client=client,                    
    collection_name="smartcharge_parent_child",
    embedding=embeddings
)
print("Parent/Child Qdrant collection initialized")


# 8 Create Parent Document Store
store = InMemoryStore()
print("Parent Document Store initialized")


# 9 Create Parent Document Retriever
retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={
        "k": 2
    }
)
print("ParentDocumentRetriever initialized")


# 10 Add documents
retriever.add_documents(documents)
print("Parent and Child documents added successfully")


# 11 Test Query
query = "Which charging station has the lowest charging cost?"

# 12 Retrieve Parent Documents
results = retriever.invoke(query)

# 13 Display Results

print("\n")
print("PARENT/CHILD RETRIEVAL")
print("\n")

for i, doc in enumerate(results, 1):

    print(f"\nResult {i}:")
    print("\nRetrieved Parent Document:")
    print(doc.page_content)


# 14 Display Metadata

print("\n")
print("RESULT METADATA")
print("\n")

for i, doc in enumerate(results, 1):

    print(f"\nResult {i} Metadata:")
    print(doc.metadata)