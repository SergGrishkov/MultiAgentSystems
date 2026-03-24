"""
Hybrid retrieval module.

Combines semantic search (vector DB) + BM25 (lexical) + cross-encoder reranking.
"""

import os
import pickle
from config import SETTINGS, VECTORSTORE_PATH, EMBEDDING_MODEL
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_openai import OpenAIEmbeddings


def get_retriever():
    # Load FAISS index
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(f"Vectorstore path not found: {VECTORSTORE_PATH}. Run ingest.py first.")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SETTINGS.openai_api_key.get_secret_value(),
    )
    faiss_store = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_store.as_retriever(search_kwargs={"k": 5})

    docs_path = os.path.join(VECTORSTORE_PATH, "documents.pkl")
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Documents pickle not found: {docs_path}. Run ingest.py first.")

    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(documents)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7],
    )

    reranker = CrossEncoderReranker(
        model="cross-encoder/msmarco-MiniLM-L-6-doc2query-msmarco-MiniLM-L-6-retrieval",
        top_n=5,
    )

    return reranker(ensemble_retriever)

