from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load the Document

loader = TextLoader("smartcharge.txt")
documents = loader.load()

print("Document Loaded:", len(documents))

#Split Document

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents)
print("Number of Chunks: ", len(chunks))

#Create Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model ="gemini-embedding-001")

#Create Vectoe Store

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)
print("Vector Store Created Successfully")

#Create retriever

retriever = vector_store.as_retriever(
    search_kwargs={"k":2}
)

# User Question
query = "What factors does SmartCharge AI consider? "

# Retrieve relevant documents
results = retriever.invoke(query)

print("\n===== RETRIEVED CONTEXT =====")

for i, doc in enumerate(results):
    print(f"\n--- Result{i + 1} ---")
    print(doc.page_content)
    
#Create LLM 
llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash")

# Create prompt

prompt = ChatPromptTemplate.from_template("""
        Answer the question using only the provided context.
        
        Context:
        {context}
        
        Question:
        {question}
        """)

#prepare context
context = "\n\n".join(
    doc.page_content for doc in results
)

#Generate answer
messages = prompt.invoke({
    "context": context,
    "qustion": query
})

response = llm.invoke(messages)

print("\n===== FINAL ANSWER =====")
print(response.content)