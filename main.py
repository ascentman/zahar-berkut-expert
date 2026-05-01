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
#   "streamlit",
# ]
# ///

import os
import pypdf
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

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
        chunk_size=2000,      # The maximum number of characters in a chunk
        chunk_overlap=400,    # The number of characters shared between adjacent chunks
        add_start_index=True, # Adds the character position in the original page to metadata
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""], # Покращені сепаратори для української мови
    )

    return text_splitter.split_documents(documents), title


def main():
    # Load variables from .env file into environment variables
    load_dotenv()

    st.set_page_config(page_title="Захар Беркут Expert", page_icon="🦅")
    st.title("🦅 Захар Беркут Expert")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "retrieval_chain" not in st.session_state:
        with st.status("Initializing Knowledge Base...", expanded=True) as status:
            st.write("Loading and chunking PDF...")

            api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

            st.write("Initializing Embeddings...")
            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2", google_api_key=api_key)

            st.write("Setting up Vector Store...")
            persist_path = "./qdrant_db"
            if os.path.exists(persist_path):
                vector_store = QdrantVectorStore.from_existing_collection(
                    embedding=embeddings,
                    path=persist_path,
                    collection_name="zahar_berkut",
                )
                title = "Захар Беркут" # Fallback title
            else:
                chunks, title = load_zahar_berkut_script()
                vector_store = QdrantVectorStore.from_documents(
                    chunks,
                    embeddings,
                    path=persist_path,
                    collection_name="zahar_berkut",
                )

            st.write("Configuring Retrieval Chain...")
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)

            # Оновлений промт: просимо бути уважнішим до деталей
            prompt = ChatPromptTemplate.from_template("""
            You are an expert on the book "Захар Беркут".
            Use the following pieces of retrieved context to answer the question.
            If the answer is partly contained, provide the best possible answer based on text in the context.
            If you don't know the answer based on the context, say that you don't know.

            Context:
            {context}

            Question: {input}

            Answer:""")

            combine_docs_chain = create_stuff_documents_chain(llm, prompt)

            # Використовуємо MultiQueryRetriever для кращого пошуку імен
            base_retriever = vector_store.as_retriever(search_kwargs={"k": 15})
            mq_retriever = MultiQueryRetriever.from_llm(
                retriever=base_retriever, llm=llm
            )

            st.session_state.retrieval_chain = create_retrieval_chain(mq_retriever, combine_docs_chain)
            status.update(label=f"Knowledge Base Ready: {title}", state="complete", expanded=False)

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if user_query := st.chat_input("Ask about Zahar Berkut..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                response = st.session_state.retrieval_chain.invoke({"input": user_query})

                # Виведення вибраних чанків у консоль для налагодження
                if "context" in response:
                    print("\n--- Retrieved Chunks (Debug) ---")
                    for i, doc in enumerate(response["context"]):
                        print(f"Chunk {i+1}:")
                        print(f"  Source Document: {doc.metadata.get('document_title', 'N/A')}")
                        print(f"  Page: {doc.metadata.get('page', 'N/A')}")
                        print(f"  Content Snippet: {doc.page_content[:300]}...\n")
                    print("----------------------------------")
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
