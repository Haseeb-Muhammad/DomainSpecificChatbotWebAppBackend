import warnings
warnings.filterwarnings("ignore")
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Define a persistent directory for the vector DB
PERSIST_DIRECTORY = "VectorDBs\\BAAIbgeLargeEn3BooksVectorDB"
DOCUMENTS_DIRECTORY  = "3books"
NEW_DOCUMENTS_DIRECTORY = "new_documents"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBEDDING_MODEL="BAAI/bge-large-en"

def load_pdfs_from_folder(folder_path):
        documents = []
        for file in os.listdir(folder_path):
            pdf_path = os.path.join(folder_path, file)
            try:
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                documents.append(docs)
            except Exception as e:
                print(f"Failed to load {pdf_path}: {e}")
            print(f"{pdf_path=}")
        return documents


def add_documents():
    docs = load_pdfs_from_folder(NEW_DOCUMENTS_DIRECTORY)
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size = CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    doc_splits = text_splitter.split_documents(docs_list)
    doc_splits = [doc for doc in doc_splits if isinstance(doc.page_content,str)]

    embedding_function = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL
    )
    vectorstore = Chroma(
        embedding_function = embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    vectorstore.add_documents(documents=doc_splits)
    vectorstore.persist()
    print("New documents Added")

def create_vector_db():
    
    docs = load_pdfs_from_folder(DOCUMENTS_DIRECTORY)
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    doc_splits = text_splitter.split_documents(docs_list)

    
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
    )
    # embedding_function = OpenAIEmbeddings()
    # Create and persist the vector store locally
    doc_splits = [doc for doc in doc_splits if isinstance(doc.page_content, str)]
    for doc in doc_splits:
        doc.page_content = str(doc.page_content)
    for i, doc in enumerate(doc_splits):
        if not isinstance(doc.page_content, str):
            print(f"[Invalid] Index: {i}, Type: {type(doc.page_content)}, Content: {repr(doc.page_content)[:200]}")
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding_function,
        persist_directory=PERSIST_DIRECTORY  # This stores the DB locally
    )
    
    # Important: Persist to disk
    vectorstore.persist()
    
    print(f"Vector database created and stored in {PERSIST_DIRECTORY}")
    return vectorstore

if __name__ == "__main__":
    # First run: Create and store the vector DB
    add_documents()