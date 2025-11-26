"""
Initialize RAG vector store from knowledge base markdown files.
Creates embeddings and stores them in ChromaDB for retrieval.
"""

import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from .env file
load_dotenv()


# Configuration
KNOWLEDGE_BASE_PATH = "data/documents/knowledge_base"
DB_NAME = "data/documents/os_tickets_vectorstore/frontier"
CHUNK_SIZE = 3000  # Increased to keep full tickets together
CHUNK_OVERLAP = 500  # Increased overlap
TOP_K_RETRIEVAL = 5


class VectorStoreInitializer:
    """Initialize and manage RAG vector store."""

    def __init__(self):
        self.documents = []
        self.chunks = []
        self.retriever = None

    def _initialize_rag(self):
        """Initialize RAG components: load documents and create vector store"""

        print("\n" + "="*70)
        print("Initializing RAG Vector Store")
        print("="*70 + "\n")

        # Load documents from Knowledge_Base folder
        print(f"📂 Loading documents from: {KNOWLEDGE_BASE_PATH}")

        # Get all markdown files directly (not folders)
        kb_path = Path(KNOWLEDGE_BASE_PATH)

        def add_metadata(doc, doc_type):
            doc.metadata["doc_type"] = doc_type
            return doc

        text_loader_kwargs = {'encoding': 'utf-8'}

        self.documents = []

        # Load markdown files from the main knowledge_base directory
        print("\n📖 Loading main knowledge base files...")
        main_loader = DirectoryLoader(
            KNOWLEDGE_BASE_PATH,
            glob="*.md",
            loader_cls=TextLoader,
            loader_kwargs=text_loader_kwargs,
            show_progress=True
        )
        main_docs = main_loader.load()
        self.documents.extend([add_metadata(doc, "knowledge_base") for doc in main_docs])
        print(f"  ✓ Loaded {len(main_docs)} files from main directory")

        # Load from subdirectories (projects_rag, quotes_rag)
        subdirs = [d for d in kb_path.iterdir() if d.is_dir()]

        for subdir in subdirs:
            doc_type = subdir.name
            print(f"\n📖 Loading {doc_type} files...")

            loader = DirectoryLoader(
                str(subdir),
                glob="*.md",
                loader_cls=TextLoader,
                loader_kwargs=text_loader_kwargs,
                show_progress=True
            )

            try:
                folder_docs = loader.load()
                self.documents.extend([add_metadata(doc, doc_type) for doc in folder_docs])
                print(f"  ✓ Loaded {len(folder_docs)} files from {doc_type}")
            except Exception as e:
                print(f"  ⚠ Error loading from {doc_type}: {e}")

        print(f"\n✓ Total documents loaded: {len(self.documents)}")

        # Split documents into chunks
        print(f"\n✂ Splitting documents into chunks...")
        print(f"  Chunk size: {CHUNK_SIZE} characters")
        print(f"  Chunk overlap: {CHUNK_OVERLAP} characters")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n---\n## Ticket", "\n---", "\n## ", "\n### ", "\n\n", "\n", " ", ""],
            length_function=len,
        )
        self.chunks = text_splitter.split_documents(self.documents)
        print(f"✓ Created {len(self.chunks)} chunks")

        # Create embeddings and vector store
        print(f"\n🔗 Creating embeddings using OpenAI...")
        embeddings = OpenAIEmbeddings()

        # Delete existing database if it exists
        db_path = Path(DB_NAME)
        if db_path.exists():
            print(f"  🗑 Clearing existing vector store at {DB_NAME}")
            try:
                # Try to delete the collection
                existing_store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
                existing_store.delete_collection()
                print(f"  ✓ Cleared existing vector store")
            except Exception as e:
                print(f"  ⚠ Could not clear existing store: {e}")
                # If deletion fails, remove the directory
                import shutil
                shutil.rmtree(DB_NAME, ignore_errors=True)
                print(f"  ✓ Removed existing directory")

        # Create new vectorstore with batching to avoid token limits
        print(f"\n💾 Creating vector store at {DB_NAME}...")
        print(f"  Processing in batches to avoid API limits...")

        # Process in batches of 100 documents
        batch_size = 100
        vectorstore = None

        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(self.chunks) + batch_size - 1) // batch_size

            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            if vectorstore is None:
                # Create initial vectorstore with first batch
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=DB_NAME
                )
            else:
                # Add subsequent batches to existing vectorstore
                vectorstore.add_documents(batch)

        # Create retriever
        self.retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RETRIEVAL}
        )

        doc_count = vectorstore._collection.count()
        print(f"✓ Vector store created with {doc_count} document chunks")

        print("\n" + "="*70)
        print("Vector Store Initialization Complete!")
        print("="*70 + "\n")

        return vectorstore

    def test_retrieval(self, query: str = "What are the main support issues?"):
        """Test the retriever with a sample query."""
        if not self.retriever:
            print("⚠ Retriever not initialized. Call _initialize_rag() first.")
            return

        print(f"\n🔍 Testing retrieval with query: '{query}'")
        results = self.retriever.invoke(query)

        print(f"\n📄 Retrieved {len(results)} documents:\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"   Type: {doc.metadata.get('doc_type', 'Unknown')}")
            print(f"   Content preview: {doc.page_content[:150]}...")
            print()


def main():
    """Main execution function."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key before running this script")
        return

    # Initialize vector store
    initializer = VectorStoreInitializer()
    vectorstore = initializer._initialize_rag()

    # Test retrieval
    print("\n" + "="*70)
    print("Testing Retrieval")
    print("="*70)

    test_queries = [
        "What support issues were reported for Piston Automotive Toledo?",
        "Tell me about MES systems projects",
        "What quotes were submitted for IAC Wauseon?"
    ]

    for query in test_queries:
        initializer.test_retrieval(query)
        print("-" * 70 + "\n")


if __name__ == "__main__":
    main()
