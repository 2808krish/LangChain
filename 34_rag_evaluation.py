import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

# 1 LOAD ENVIRONMENT VARIABLES
load_dotenv()

# 2 LOAD DOCUMENT
loader = TextLoader("smartcharge.txt")
documents = loader.load()
print("Documents Loaded:", len(documents))

# 3 SPLIT DOCUMENT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print("Number of Chunks:", len(chunks))

# 4 INITIALIZE EMBEDDINGS
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
print("Embedding model initialized")

# 5 CONNECT TO QDRANT
client = QdrantClient(path="./qdrant_data")

vector_store = QdrantVectorStore(
    client=client,
    collection_name="smartcharge",
    embedding=embeddings
)
print("Qdrant Vector Store connected")

# 6 CREATE RETRIEVER
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)

print("Retriever initialized")

# 7. INITIALIZE GEMINI
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
print("Gemini initialized successfully")

# 8 RAG ANSWER PROMPT
answer_prompt = ChatPromptTemplate.from_template(
    """
You are SmartCharge AI.

Answer the user's question using ONLY the provided context.

Do not invent information.

If the context does not contain enough information,
say that the information is not available.

Question:
{question}

Context:
{context}

Answer:
"""
)

answer_chain = ( answer_prompt | llm | StrOutputParser())

# 9 EVALUATION DATASET
evaluation_data = [
    {
        "question": "What factors does SmartCharge AI consider?",
        "expected_answer": (
            "SmartCharge AI considers battery percentage, "
            "current location, destination, distance, "
            "charging speed, charging cost, station availability, "
            "station reliability, and route optimization."
        )
    },
    {
        "question": "What is the goal of SmartCharge AI?",
        "expected_answer": (
            "The goal is to recommend where an EV driver "
            "should charge and explain why."
        )
    },
    {
        "question": "Can SmartCharge AI estimate charging cost?",
        "expected_answer": (
            "Yes, SmartCharge AI can estimate charging cost."
        )
    }
]

# 10 EVALUATION PROMPT

evaluation_prompt = ChatPromptTemplate.from_template(
    """
You are evaluating a RAG system.

Evaluate the answer using the question,
retrieved context, and expected answer.

Give a score from 1 to 5 for each metric.

1. Context Relevance:
Is the retrieved context relevant to the question?

2. Answer Relevance:
Does the generated answer directly answer the question?

3. Answer Correctness:
Does the generated answer agree with the expected answer?

Return ONLY this format:

Context Relevance: X/5
Answer Relevance: X/5
Answer Correctness: X/5

Question:
{question}

Expected Answer:
{expected_answer}

Retrieved Context:
{context}

Generated Answer:
{answer}
"""
)

evaluation_chain = ( evaluation_prompt | llm | StrOutputParser())

# 11 RUN EVALUATION

print("\n")
print("========================================")
print("RAG EVALUATION")
print("========================================")


for i, item in enumerate(evaluation_data, 1):

    question = item["question"]
    expected_answer = item["expected_answer"]

    # RETRIEVE CONTEXT

    retrieved_documents = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_documents
    )

    # GENERATE ANSWER

    answer = answer_chain.invoke(
        {
            "question": question,
            "context": context
        }
    )

    # EVALUATE

    evaluation = evaluation_chain.invoke(
        {
            "question": question,
            "expected_answer": expected_answer,
            "context": context,
            "answer": answer
        }
    )

    # DISPLAY RESULTS

    print("\n")
    print("----------------------------------------")
    print(f"Evaluation {i}")
    print("----------------------------------------")

    print("\nQuestion:")
    print(question)

    print("\nExpected Answer:")
    print(expected_answer)

    print("\nRetrieved Context:")
    print(context)

    print("\nGenerated Answer:")
    print(answer)

    print("\nEvaluation:")
    print(evaluation)

# 12. CLOSE QDRANT

client.close()
print("\n")
print("RAG Evaluation completed successfully.")