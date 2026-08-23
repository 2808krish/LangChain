import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. CHECK API KEY
# ============================================================

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Google API key not found.\n"
        "Set GOOGLE_API_KEY or GEMINI_API_KEY before running the program."
    )


# ============================================================
# 2. LOAD DOCUMENT
# ============================================================

loader = TextLoader("smartcharge copy.txt")

documents = loader.load()

print("Document Loaded:", len(documents))


# ============================================================
# 3. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Number of Chunks:", len(chunks))


# ============================================================
# 4. CREATE EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)


# ============================================================
# 5. CREATE VECTOR STORE
# ============================================================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="smartcharge_collection"
)

print("Vector Store created successfully")


# ============================================================
# 6. CREATE RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# ============================================================
# 7. CREATE LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)


# ============================================================
# 8. QUESTION REWRITING PROMPT
# ============================================================

rewrite_prompt = ChatPromptTemplate.from_template(
    """
Rewrite the user's question into a standalone question.

Conversation history:
{chat_history}

Current question:
{question}

Return ONLY the rewritten question.
"""
)


# ============================================================
# 9. RAG PROMPT
# ============================================================

rag_prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant for SmartCharge AI.

Answer the user's question using ONLY the provided context.

If the answer is not present in the context, say:
"I don't have enough information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# 10. CHAT HISTORY
# ============================================================

chat_history = []


# ============================================================
# 11. CHAT FUNCTION
# ============================================================

def chat(user_question):

    print("\n==============================")
    print("USER:", user_question)
    print("==============================")


    # --------------------------------------------------------
    # STEP 1: Rewrite the question
    # --------------------------------------------------------

    rewrite_messages = rewrite_prompt.invoke({
        "chat_history": chat_history,
        "question": user_question
    })

    rewrite_response = llm.invoke(rewrite_messages)

    rewritten_question = rewrite_response.content

    print("\nRewritten Question:")
    print(rewritten_question)


    # --------------------------------------------------------
    # STEP 2: Retrieve relevant documents
    # --------------------------------------------------------

    results = retriever.invoke(rewritten_question)

    print("\n===== RETRIEVED CONTEXT =====")

    context_parts = []

    for i, doc in enumerate(results):

        print(f"\n--- Result {i + 1} ---")
        print(doc.page_content)

        context_parts.append(doc.page_content)


    # --------------------------------------------------------
    # STEP 3: Combine retrieved documents
    # --------------------------------------------------------

    context = "\n\n".join(context_parts)


    # --------------------------------------------------------
    # STEP 4: Create RAG prompt
    # --------------------------------------------------------

    rag_messages = rag_prompt.invoke({
        "context": context,
        "question": rewritten_question
    })


    # --------------------------------------------------------
    # STEP 5: Ask Gemini
    # --------------------------------------------------------

    response = llm.invoke(rag_messages)

    answer = response.content


    # --------------------------------------------------------
    # STEP 6: Save conversation history
    # --------------------------------------------------------

    chat_history.append(
        f"User: {user_question}"
    )

    chat_history.append(
        f"Assistant: {answer}"
    )


    # --------------------------------------------------------
    # STEP 7: Return answer
    # --------------------------------------------------------

    return answer


# ============================================================
# 12. TEST THE CONVERSATIONAL RAG
# ============================================================

print("\n\n===== CONVERSATIONAL RAG =====")


answer = chat(
    "What is SmartCharge AI?"
)

print("\nAI:", answer)


answer = chat(
    "What does it recommend?"
)

print("\nAI:", answer)


answer = chat(
    "Why is it different from a normal charging station finder?"
)

print("\nAI:", answer)