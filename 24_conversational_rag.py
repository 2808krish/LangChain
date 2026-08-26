from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# 1 LOAD DOCUMENT

loader = TextLoader("smartcharge.txt")
documents = loader.load()

print("Document Loaded:", len(documents))

# 2 SPLIT DOCUMENT INTO CHUNKS

text_splitter= RecursiveCharacterTextSplitter(
    chunk_size= 500,
    chunk_overlap = 50
)
chunks = text_splitter.split_documents(documents)
print("Number of Chunks:", len(chunks))

# 3 CREATE EMBEDDING

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
print("Embedding Model Created Successfully")

# 4 CREATE QDRANT VECTOR STORE

vector_store = QdrantVectorStore.from_documents(
    documents= chunks,
    embedding= embeddings,
    location= ":memory:",
    collection_name= "smartcharge_rag"   
)

print("Qdrant Vector Store Created Successfully")

# 5 CREATE RETRIEVER

retriever = vector_store.as_retriever(
    search_kwargs = {"k":3}
)


# 6 CREATE LLM

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

# 7 QUESTION REWRITING PROMPT

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You rewrite a user's question into a standalone question.

Use the conversation history when necessary.

If the question is already standalone, return it unchanged.

Return ONLY the rewritten question.
"""
    ),
    (
        "human",
        """
Conversation history:
{history}

User question:
{question}
"""
    )
])

# 8 QUESTION REWRITING CHAIN

rewrite_chain = rewrite_prompt | llm | StrOutputParser()

# 9 ANSWER PROMPT

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an assistant for SmartCharge AI.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I don't have enough information in the provided documents."

Do not invent information.

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])


# 11. ANSWER CHAIN

answer_chain = answer_prompt | llm | StrOutputParser()

# 12. CHAT HISTORY

chat_history = []

# 13. CHAT FUNCTION

def chat(user_question):

    print("\n===== SESSION =====")
    print("User:", user_question)

    # Step A: Rewrite question
  
    history_text = "\n".join(
        [
            f"User: {user_msg}\nAI: {ai_msg}"
            for user_msg, ai_msg in chat_history
        ]
    )

    rewritten_question = rewrite_chain.invoke({
        "history": history_text,
        "question": user_question
    })

    print("Rewritten Question:", rewritten_question)

    # --------------------------------------------------------
    # Step B: Retrieve documents from Qdrant
    # --------------------------------------------------------

    results = retriever.invoke(rewritten_question)

    print("\n===== RETRIEVED CONTEXT =====")

    context = ""

    for i, doc in enumerate(results):

        print(f"\n--- Result {i + 1} ---")
        print(doc.page_content)

        context += doc.page_content + "\n\n"

    # --------------------------------------------------------
    # Step C: Generate answer
    # --------------------------------------------------------

    answer = answer_chain.invoke({
        "context": context,
        "question": rewritten_question
    })

    # --------------------------------------------------------
    # Step D: Save conversation
    # --------------------------------------------------------

    chat_history.append(
        (user_question, answer)
    )

    return answer


# ============================================================
# 14. TEST CONVERSATION
# ============================================================

print("\n===== SMARTCHARGE AI QDRANT CONVERSATIONAL RAG =====")

print("\nAI:", chat("What is SmartCharge AI?"))

print("\nAI:", chat("What does it recommend?"))

print("\nAI:", chat("Why is it different from a normal charging station finder?"))