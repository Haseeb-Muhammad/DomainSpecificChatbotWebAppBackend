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


class VectorDatabaseManager:
    """
    A comprehensive class for managing vector databases with PDF documents
    """
    
    def __init__(self, documents_directory, model_name="BAAI/bge-small-en", 
                 collection_name="rag-chroma", chunk_size=1500, chunk_overlap=200):
        """
        Initialize the Vector Database Manager
        
        Args:
            documents_directory (str): Path to the directory containing PDF files
            model_name (str): Name of the embedding model to use
            collection_name (str): Name of the Chroma collection
            chunk_size (int): Size of text chunks for splitting
            chunk_overlap (int): Overlap between chunks
        """
        self.documents_directory = documents_directory
        self.model_name = model_name
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Generate persist directory based on model and document directory
        self.persist_directory = f"VectorDBs/{os.path.basename(model_name)}_{os.path.basename(documents_directory)}"
        
        # Initialize embedding function
        self.embedding_function = HuggingFaceEmbeddings(model_name=model_name)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Vectorstore will be loaded/created when needed
        self._vectorstore = None
        
        print(f"VectorDatabaseManager initialized")
        print(f"Documents Directory: {self.documents_directory}")
        print(f"Persist Directory: {self.persist_directory}")
        print(f"Model: {self.model_name}")
    
    @property
    def vectorstore(self):
        """
        Lazy loading property for vectorstore
        """
        if self._vectorstore is None:
            if self.check_database_exists():
                self._vectorstore = self.load_existing_database()
            else:
                print("No existing database found. Use create_or_update_database() to create one.")
        return self._vectorstore
    
    def check_database_exists(self):
        """
        Check if vector database already exists
        
        Returns:
            bool: True if database exists, False otherwise
        """
        return os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0
    
    def load_existing_database(self):
        """
        Load existing vector database
        
        Returns:
            Chroma: Loaded vectorstore instance
        """
        vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )
        
        print(f"Loaded existing vector database from {self.persist_directory}")
        return vectorstore
    
    def load_pdfs_from_folder(self, limit=None, combine_pages=True):
        """
        Load PDFs from the configured folder with optional limit
        
        Args:
            limit (int): Maximum number of PDFs to load (None for all)
            combine_pages (bool): Whether to combine all pages of a PDF before splitting
        
        Returns:
            list: List of loaded documents with metadata
        """
        documents = []
        files = os.listdir(self.documents_directory)
        
        if limit:
            files = files[:limit]
        
        for file in files:
            if file.lower().endswith('.pdf'):
                print(f"Loading: {file}")
                pdf_path = os.path.join(self.documents_directory, file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    docs = loader.load()
                    
                    if combine_pages:
                        # Combine all pages into a single document
                        combined_text = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Create a single document with combined content
                        combined_doc = docs[0]  # Use first doc as base
                        combined_doc.page_content = combined_text
                        combined_doc.metadata['pdf_name'] = file
                        combined_doc.metadata['source_file'] = pdf_path
                        combined_doc.metadata['total_pages'] = len(docs)
                        combined_doc.metadata['page'] = 'combined'
                        
                        documents.append(combined_doc)
                    else:
                        # Keep original per-page approach
                        for doc in docs:
                            doc.metadata['pdf_name'] = file
                            doc.metadata['source_file'] = pdf_path
                        documents.extend(docs)
                        
                except Exception as e:
                    print(f"Failed to load {pdf_path}: {e}")
        
        return documents
    
    def get_existing_pdf_names(self):
        """
        Get list of PDF names already in the database
        
        Returns:
            set: Set of PDF names currently in database
        """
        if not self.check_database_exists():
            return set()
        
        try:
            vectorstore = self.vectorstore
            all_docs = vectorstore.get()
            
            if not all_docs or 'metadatas' not in all_docs:
                return set()
            
            existing_pdf_names = {
                meta.get('pdf_name', '') for meta in all_docs['metadatas'] 
                if meta and 'pdf_name' in meta
            }
            
            # Remove empty strings
            existing_pdf_names.discard('')
            return existing_pdf_names
            
        except Exception as e:
            print(f"Could not retrieve existing PDF names: {e}")
            return set()
    
    def create_or_update_database(self, new_pdfs_only=True, pdf_limit=None):
        """
        Create new vector DB or append to existing one
        
        Args:
            new_pdfs_only (bool): If True and DB exists, only process new PDFs not already in DB
            pdf_limit (int): Limit number of PDFs to process (None for all)
        
        Returns:
            Chroma: The vectorstore instance
        """
        db_exists = self.check_database_exists()
        
        if db_exists:
            print("Vector database exists. Loading existing database...")
            self._vectorstore = self.load_existing_database()
            
            # Get existing PDF names if we want to skip already processed files
            existing_pdf_names = set()
            if new_pdfs_only:
                existing_pdf_names = self.get_existing_pdf_names()
                print(f"Found {len(existing_pdf_names)} existing PDFs in database")
            
            # Load new documents
            all_docs = self.load_pdfs_from_folder(pdf_limit)
            
            # Filter out already processed PDFs if requested
            if new_pdfs_only and existing_pdf_names:
                new_docs = [
                    doc for doc in all_docs 
                    if doc.metadata.get('pdf_name', '') not in existing_pdf_names
                ]
                print(f"Found {len(new_docs)} new documents to add (filtered from {len(all_docs)} total)")
            else:
                new_docs = all_docs
                print(f"Processing {len(new_docs)} documents")
            
            if not new_docs:
                print("No new documents to add.")
                return self._vectorstore
            
            # Process and add new documents
            doc_splits = self._process_documents(new_docs)
            
            if doc_splits:
                self._vectorstore.add_documents(doc_splits)
                self._vectorstore.persist()
                print(f"Added {len(doc_splits)} new document chunks to existing database")
            
        else:
            print("Creating new vector database...")
            
            # Load documents
            docs = self.load_pdfs_from_folder(pdf_limit)
            
            if not docs:
                print("No documents found to process.")
                return None
            
            # Process documents
            doc_splits = self._process_documents(docs)
            
            if not doc_splits:
                print("No valid document splits created.")
                return None
            
            # Create new vector store
            self._vectorstore = Chroma.from_documents(
                documents=doc_splits,
                collection_name=self.collection_name,
                embedding=self.embedding_function,
                persist_directory=self.persist_directory
            )
            
            # Persist to disk
            self._vectorstore.persist()
            print(f"Vector database created and stored in {self.persist_directory}")
        
        return self._vectorstore
    
    def _process_documents(self, documents):
        """
        Internal method to process and validate documents
        
        Args:
            documents (list): List of documents to process
        
        Returns:
            list: List of processed document splits
        """
        # Split documents
        doc_splits = self.text_splitter.split_documents(documents)
        
        # Clean and validate documents
        doc_splits = [doc for doc in doc_splits if isinstance(doc.page_content, str)]
        for doc in doc_splits:
            doc.page_content = str(doc.page_content)
        
        # Validate document content
        invalid_docs = []
        for i, doc in enumerate(doc_splits):
            if not isinstance(doc.page_content, str):
                invalid_docs.append(i)
                print(f"[Invalid] Index: {i}, Type: {type(doc.page_content)}, Content: {repr(doc.page_content)[:200]}")
        
        if invalid_docs:
            print(f"Found {len(invalid_docs)} invalid documents that will be skipped")
        
        return doc_splits
    
    def delete_documents_by_pdf_name(self, pdf_name):
        """
        Delete all documents from a specific PDF file
        
        Args:
            pdf_name (str): Name of the PDF file (e.g., 'document.pdf')
        
        Returns:
            int: Number of documents deleted
        """
        if not self.check_database_exists():
            print("No vector database found.")
            return 0
        
        vectorstore = self.vectorstore
        
        try:
            # Get all documents
            all_docs = vectorstore.get()
            
            if not all_docs or 'metadatas' not in all_docs:
                print("No documents found in database.")
                return 0
            
            # Find document IDs that match the PDF name
            ids_to_delete = []
            for i, metadata in enumerate(all_docs['metadatas']):
                if metadata and metadata.get('pdf_name') == pdf_name:
                    ids_to_delete.append(all_docs['ids'][i])
            
            if not ids_to_delete:
                print(f"No documents found for PDF: {pdf_name}")
                return 0
            
            # Delete the documents
            vectorstore.delete(ids=ids_to_delete)
            vectorstore.persist()
            
            print(f"Deleted {len(ids_to_delete)} documents from PDF: {pdf_name}")
            return len(ids_to_delete)
            
        except Exception as e:
            print(f"Error deleting documents for PDF {pdf_name}: {e}")
            return 0
    
    def list_pdf_names_in_database(self):
        """
        List all PDF names currently in the database
        
        Returns:
            set: Set of PDF names in the database
        """
        if not self.check_database_exists():
            print("No vector database found.")
            return set()
        
        return self.get_existing_pdf_names()
    
    def get_database_statistics(self):
        """
        Get comprehensive statistics about the vector database
        
        Returns:
            dict: Database statistics including document count, PDF count, and PDF names
        """
        if not self.check_database_exists():
            return {
                "exists": False, 
                "total_documents": 0, 
                "pdf_count": 0, 
                "pdf_names": [],
                "persist_directory": self.persist_directory
            }
        
        try:
            vectorstore = self.vectorstore
            all_docs = vectorstore.get()
            pdf_names = self.list_pdf_names_in_database()
            
            return {
                "exists": True,
                "total_documents": len(all_docs['ids']) if all_docs else 0,
                "pdf_count": len(pdf_names),
                "pdf_names": sorted(list(pdf_names)),
                "persist_directory": self.persist_directory,
                "model_name": self.model_name,
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {
                "exists": True, 
                "error": str(e),
                "persist_directory": self.persist_directory
            }
    
    def search_documents(self, query, k=5):
        """
        Search for similar documents in the vector database
        
        Args:
            query (str): Search query
            k (int): Number of results to return
        
        Returns:
            list: List of similar documents
        """
        if not self.check_database_exists():
            print("No vector database found. Create database first.")
            return []
        
        try:
            vectorstore = self.vectorstore
            results = vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def reset_database(self):
        """
        Reset/delete the entire vector database
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
                self._vectorstore = None
                print(f"Database reset successfully. Deleted: {self.persist_directory}")
                return True
            else:
                print("No database found to reset.")
                return True
        except Exception as e:
            print(f"Error resetting database: {e}")
            return False
    
    def __str__(self):
        """String representation of the VectorDatabaseManager"""
        stats = self.get_database_statistics()
        return f"""VectorDatabaseManager:
  Documents Directory: {self.documents_directory}
  Persist Directory: {self.persist_directory}
  Model: {self.model_name}
  Collection: {self.collection_name}
  Database Exists: {stats['exists']}
  Total Documents: {stats.get('total_documents', 0)}
  PDF Count: {stats.get('pdf_count', 0)}"""


def main():
    """Example usage of the VectorDatabaseManager class"""
    
    # Initialize the manager
    documents_dir = "/home/haseeb/Desktop/NCAI/Datasets/AI_books"
    db_manager = VectorDatabaseManager(
        documents_directory=documents_dir,
        model_name="BAAI/bge-small-en",
        collection_name="rag-chroma"
    )
    
    print("=== Vector Database Manager Info ===")
    print(db_manager)
    
    # Create or update database
    print("\n=== Creating/Updating Vector Database ===")
    vectorstore = db_manager.create_or_update_database(new_pdfs_only=True, pdf_limit=2)
    
    if vectorstore:
        # Get database statistics
        print("\n=== Database Statistics ===")
        stats = db_manager.get_database_statistics()
        print(f"Total documents: {stats.get('total_documents', 0)}")
        print(f"Number of PDFs: {stats.get('pdf_count', 0)}")
        print(f"PDF names: {stats.get('pdf_names', [])}")
        
        # Search example
        print("\n=== Search Example ===")
        search_results = db_manager.search_documents("artificial intelligence", k=3)
        print(f"Found {len(search_results)} search results")
        
        # List PDFs in database
        print("\n=== PDFs in Database ===")
        pdf_names = db_manager.list_pdf_names_in_database()
        print(f"PDFs: {list(pdf_names)}")
        
        # Example: Delete documents from a specific PDF
        if pdf_names:
            print("\n=== Delete Example ===")
            first_pdf = list(pdf_names)[0]
            print(f"Deleting documents from: {first_pdf}")
            deleted_count = db_manager.delete_documents_by_pdf_name(first_pdf)
            print(f"Deleted {deleted_count} documents")
            
            # Show updated stats
            print("\n=== Updated Statistics ===")
            updated_stats = db_manager.get_database_statistics()
            print(f"Total documents: {updated_stats.get('total_documents', 0)}")
            print(f"Number of PDFs: {updated_stats.get('pdf_count', 0)}")


if __name__ == "__main__":
    main()