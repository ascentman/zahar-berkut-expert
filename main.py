# /// script
# dependencies = [
#   "langchain-community",
#   "pypdf",
#   "langchain-text-splitters",
#   "langchain-google-genai",
#   "python-dotenv",
#   "langchain-qdrant",
#   "qdrant-client",
#   "langchain",
# ]
# ///

import os
import pypdf
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def load_zahar_berkut_script():
    # Use a path relative to the script's directory for portability
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "full-text.pdf")

    # Extract title from PDF metadata using pypdf
    reader = pypdf.PdfReader(pdf_path)
    title = reader.metadata.title or "Захар Беркут"

    # Initialize the PDF loader
    loader = PyPDFLoader(pdf_path)

    # Load the PDF (one LangChain Document per page)
    documents = loader.load()

    # Add the document title to each page's metadata before splitting
    for doc in documents:
        doc.metadata["document_title"] = title

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # The maximum number of characters in a chunk
        chunk_overlap=200,    # The number of characters shared between adjacent chunks
        add_start_index=True, # Adds the character position in the original page to metadata
        separators=["\n\n", "\n", ". ", " ", ""], # Tries to keep paragraphs, then lines, then sentences whole
    )

    return text_splitter.split_documents(documents), title


def main():
    # Load variables from .env file into environment variables
    load_dotenv()

    # Load and split the script
    chunks, title = load_zahar_berkut_script()

    # Script metadata
    script = {"title": title}

    print(f"Loaded and split script for {script['title']} into {len(chunks)}")

    # Initialize Gemini Embeddings
    # Ensure you have the GOOGLE_API_KEY environment variable set
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

    print("Gemini embeddings model initialized.")

    # Create Qdrant vector store and persist it to a local directory
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        path="./qdrant_db",
        collection_name="zahar_berkut",
    )

    print(f"Successfully stored {len(chunks)} vectors in local Qdrant DB at ./qdrant_db")

    # Initialize the LLM (using gemini-2.0-flash from your available models)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Configure the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an expert on the book "Захар Беркут".
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer based on the context, say that you don't know.

    Context:
    {context}

    Question: {input}

    Answer:""")

    # Set up the retrieval chain with k=15 chunks
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 15})
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("\n--- Q&A Loop Started ---")
    print("Ask me anything about Zahar Berkut (type 'exit' to quit)")

    while True:
        user_input = input("\nYour Question: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        response = retrieval_chain.invoke({"input": user_input})
        print(f"\nAnswer: {response['answer']}")


if __name__ == "__main__":
    main()
