import os

import pypdf
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_logger import InstrumentedRetriever, RAGLogger


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
    """Initialize the retrieval chain without any Streamlit dependency."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=api_key)
    persist_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_db")

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
        chunks, _ = load_zahar_berkut_chunks()
        vector_store = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            path=persist_path,
            collection_name="zahar_berkut",
        )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)

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
    retriever = InstrumentedRetriever(vector_store=vector_store, llm=llm, k=12)
    logger = RAGLogger()
    chain = create_retrieval_chain(retriever, combine_docs_chain)

    return chain, retriever, logger
