# rag_chain.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.services.openai import GPT4LLM as gpt4_answer
from backend.services.anthropic import ClaudeLLM as claude_answer
from backend.services.open_source import LocalLLM as open_source_answer

import pickle
import os

index_path = "retrieval/index"
chunks_path = "retrieval/cache_data/chunks.pkl"

def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def load_chunks():
    with open(chunks_path, "rb") as f:
        return pickle.load(f)

def retrieve_context(question: str, k: int = 3):
    retriever = load_retriever().as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(question)

def format_context(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def main():
    question = input("❓ Enter your question: ")
    context_docs = retrieve_context(question)
    context = format_context(context_docs)

    print("\n🔎 Retrieved Context:")
    for i, doc in enumerate(context_docs):
        source = doc.metadata.get('source', 'No source info')
        print(f"[{i+1}] {source}\n{doc.page_content[:300]}...\n")

    prompt = f"Context:\n{context}\n\nQuestion: {question}"

    #print("\n🤖 Claude Answer:")
    #try:
    #    claude_llm = claude_answer()
    #    print(claude_llm.generate(prompt))
    #except Exception as e:
    #    print(f"[Claude ERROR] {e}")

    print("\n🤖 Mistral Answer (local Falcon 1B):")
    try:
        mistral_llm = open_source_answer()
        result = mistral_llm.generate([prompt])
        print(result.generations[0][0].text)
    except Exception as e:
        print(f"[Mistral ERROR] {e}")

if __name__ == "__main__":
    main()