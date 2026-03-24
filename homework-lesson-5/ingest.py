"""
Knowledge ingestion pipeline.

Loads documents from data/ directory, splits into chunks,
generates embeddings, and saves the index to disk.

Usage: python ingest.py
"""

import os
from glob import glob
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from config import SETTINGS, VECTORSTORE_PATH, EMBEDDING_MODEL


def ingest():
    pdf_files = glob("./data/*.pdf")
    if not pdf_files:
        print("No PDF files found in ./data/. Please add at least one PDF for ingestion.")
        return

    all_documents = []
    for pdf_path in pdf_files:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        all_documents.extend(docs)

    print(f"📄 Loaded {len(all_documents)} pages from {len(pdf_files)} files")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_documents)
    print(f"✂️  Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SETTINGS.openai_api_key.get_secret_value(),
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print(f"🗃️  FAISS index built!")

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    docs_path = os.path.join(VECTORSTORE_PATH, "documents.pkl")
    with open(docs_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"💾 Index and chunk data saved to {VECTORSTORE_PATH}/")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0,
        api_key=SETTINGS.openai_api_key.get_secret_value(),
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the provided context.\n\nContext:\n{context}"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\n🔍 Testing query: 'What is retrieval-augmented generation?'")
    result = qa_chain.invoke({"input": "What is retrieval-augmented generation?"})
    print(f"\n📝 Answer: {result.get('answer', 'No answer')}")
    print(f"\n📚 Sources:")
    for doc in result.get('context', []):
        print(f"   - Page {doc.metadata.get('page', '?')}")


if __name__ == "__main__":
    ingest()
