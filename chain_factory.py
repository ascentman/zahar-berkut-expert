import os

import pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from rag_logger import InstrumentedRetriever, RAGLogger

# Multilingual cross-encoder trained on mMARCO (43 languages including Ukrainian).
# ~130MB download on first run, cached in ~/.cache/huggingface/ afterwards.
_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def load_zahar_berkut_chunks():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "full-text.pdf")

    reader = pypdf.PdfReader(pdf_path)
    title = reader.metadata.title or "Захар Беркут"

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["document_title"] = title

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    return text_splitter.split_documents(documents), title


def create_retrieval_chain_headless(api_key: str):
    """Initialize the retrieval chain without any Streamlit dependency.

    Retrieval pipeline:
      1. Multi-query: LLM generates 2-3 sub-queries per question
      2. Hybrid search: vector (Qdrant cosine) + BM25 (keyword) per sub-query
      3. RRF merge of all candidates
      4. Cross-encoder rerank of top-25 → final top-12
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=api_key)
    persist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_db")

    # Always load chunks — needed for BM25 (fast: PDF parse only, no API calls)
    chunks, _ = load_zahar_berkut_chunks()

    vector_store = None
    if os.path.exists(persist_path) and os.path.isdir(persist_path):
        try:
            vector_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings,
                path=persist_path,
                collection_name="zahar_berkut",
            )
        except RuntimeError as e:
            if "already accessed" in str(e):
                raise RuntimeError(
                    "Qdrant DB is locked by another process (likely Streamlit). "
                    "Stop Streamlit before running the eval harness."
                ) from e
            vector_store = None
        except Exception:
            vector_store = None

    if vector_store is None:
        vector_store = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            path=persist_path,
            collection_name="zahar_berkut",
        )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)

    # BM25 index over the same chunks — complements vector search for exact keyword matches
    # (proper nouns like "Бурунда", "Тухольщина" that embeddings may miss)
    bm25_retriever = BM25Retriever.from_documents(chunks, k=20)

    # Multilingual cross-encoder — scores (query, chunk) pairs together for precise reranking
    # First run downloads ~130MB to ~/.cache/huggingface/
    cross_encoder = CrossEncoder(_RERANKER_MODEL)

    prompt = ChatPromptTemplate.from_template("""
    You are an expert on the book "Захар Беркут".
    Use the following pieces of retrieved context to answer the question.
    If the answer is partly contained, provide the best possible answer based on text in the context.
    If you don't know the answer based on the context, say that you don't know.
    Reference the page numbers from the context that you used in your answer.

    Context:
    {context}

    Question: {input}

    Answer:""")

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = InstrumentedRetriever(
        vector_store=vector_store,
        llm=llm,
        k=12,
        bm25_retriever=bm25_retriever,
        cross_encoder=cross_encoder,
        rerank_candidates=25,
    )
    logger = RAGLogger()
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    return chain, retriever, logger
