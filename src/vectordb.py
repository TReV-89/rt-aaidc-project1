import os
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings
        """
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", "rag_documents"
        )
        self.embedding_model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="./chroma_db")

        # Load embedding model
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "description": "RAG document collection",
                "hnsw:space": "cosine",
                "hnsw:batch_size": 10000,
            },
        )

        print(f"Vector database initialized with collection: {self.collection_name}")

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """
        Simple text chunking by splitting on spaces and grouping into chunks.

        Args:
            text: Input text to chunk
            chunk_size: Approximate number of characters per chunk

        Returns:
            List of text chunks
        """
        # TODO: Implement text chunking logic
        # You have several options for chunking text - choose one or experiment with multiple:
        #
        # OPTION 1: Simple word-based splitting
        #   - Split text by spaces and group words into chunks of ~chunk_size characters
        #   - Keep track of current chunk length and start new chunks when needed
        #
        # OPTION 2: Use LangChain's RecursiveCharacterTextSplitter
        #   - from langchain_text_splitters import RecursiveCharacterTextSplitter
        #   - Automatically handles sentence boundaries and preserves context better
        #
        # OPTION 3: Semantic splitting (advanced)
        #   - Split by sentences using nltk or spacy
        #   - Group semantically related sentences together
        #   - Consider paragraph boundaries and document structure
        #
        # Feel free to try different approaches and see what works best!

        chunks = []
        # Your implementation
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = text_splitter.split_text(text)

        return chunks

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        # TODO: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user

        print(f"Processing {len(documents)} documents...")
        # Your implementation here
        doc_id = 0
        for document in documents:
            chunked_content = self.chunk_text(document.page_content)
            texts = [doc for doc in chunked_content]
            metadata = [
                {"source": str(document.metadata), "chunk_id": i, "doc_id": doc_id}
                for i in range(len(texts))
            ]
            id = [f"doc_{doc_id}_chunk_{i}" for i in range(len(texts))]
            embeddings = self.embedding_model.encode(texts).tolist()
            self.collection.add(
                ids=id,
                documents=texts,
                metadatas=metadata,
                embeddings=embeddings,
            )
            doc_id += 1
            print(f"Added {len(chunked_content)} chunks from document ID: {id}")

        print("Documents added to vector database")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # TODO: Implement similarity search logic
        # HINT: Use self.embedding_model.encode([query]) to create query embedding
        # HINT: Convert the embedding to appropriate format for your vector database
        # HINT: Use your vector database's search/query method with the query embedding and n_results
        # HINT: Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        # HINT: Handle the case where results might be empty

        # Your implementation here
        query_vector = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_vector,
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # Handle empty results
        if not results["documents"] or not results["documents"][0]:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": [],
            }

        relevant_chunks = []
        for i, doc in enumerate(results["documents"][0]):
            relevant_chunks.append(
                {
                    "documents": doc,
                    "metadatas": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                }
            )
        return {
            "documents": [doc for doc in results["documents"][0]],
            "metadatas": [meta for meta in results["metadatas"][0]],
            "distances": [dist for dist in results["distances"][0]],
            "ids": [meta['doc_id'] for meta in results["metadatas"][0]],
        }
