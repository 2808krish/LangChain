from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# documents with metadata
documents =[
    Document(
        page_content="Station A is located in Ahmedabad and provides DC fast charging.",
        metadata={
            "city": "Ahmedabad",
            "charger_type": "DC Fast",
            "station": "Station A"
        }
    ),
    
     Document(
        page_content="Station B is located in Mumbai and provides AC charging.",
        metadata={
            "city": "Mumbai",
            "charger_type": "AC",
            "station": "Station B"
        }
    ),

    Document(
        page_content="Station C is located in Ahmedabad and provides AC charging.",
        metadata={
            "city": "Ahmedabad",
            "charger_type": "AC",
            "station": "Station C"
        }
    ),

    Document(
        page_content="Station D is located in Delhi and provides DC fast charging.",
        metadata={
            "city": "Delhi",
            "charger_type": "DC Fast",
            "station": "Station D"
        }
    )
]

print("Documents created:", len(documents))

# create embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)
print("Embedding Model Initialized")

#Creating chroma vector store
vector_store = Chroma.from_documents(
    documents=documents,
    embedding= embeddings,
    collection_name="smartcharge_metadata_demo",
    persist_directory="./chroma_metadata_demo"
)
print("Vecotr store created successfully")

# create retriever 
retriever= vector_store.as_retriever(
    search_kwargs={"k":4}
)

#normal retrieval

print("\n===== NORMAL RETRIEVAL =====")

results = retriever.invoke("charging station")

for i,doc in enumerate(results):
    print(f"\n--- Result{i+1} ---")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    
# METADATA FILTERING 
print("\n===== AHMEDABAD ONLY =====")
results = vector_store.similarity_search(
    "charging station",
    k=4,
    filter={
        "city": "Ahmedabad"
    }
)

for i, doc in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("Content:", doc.page_content)

    print("Metadata:", doc.metadata)

#METADATA + SEMANTIC SEARCH 
print("\n===== AHMEDABAD + DC FAST =====")

results = vector_store.similarity_search(
    "fast charging station",
    k=4,
    filter={
        "$and": [
            {"city": "Ahmedabad"},
            {"charger_type": "DC Fast"}
        ]
    }
)

for i, doc in enumerate(results):

    print(f"\n--- Result {i + 1} ---")

    print("Content:", doc.page_content)

    print("Metadata:", doc.metadata)