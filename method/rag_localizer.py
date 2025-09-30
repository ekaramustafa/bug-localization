import os
import json
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dataset.utils import get_logger

logger = get_logger(__name__)

class RAGLocalizer:
    def __init__(self, collection_name="bug_localization", chunk_size=1000, chunk_overlap=150):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self._init_collection()
    
    def _init_collection(self):
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
            )
            logger.info(f"Initialized collection: {self.collection_name}")
        except Exception as e:
            logger.info(f"Collection {self.collection_name} already initialized or error: {e}")
    
    def _get_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    
    def _chunk_file_content(self, file_path: str, content: str) -> List[Dict[str, Any]]:
        chunks = self.text_splitter.split_text(content)
        chunk_data = []
        
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "file_path": file_path,
                "chunk_id": f"{file_path}_{i}",
                "content": chunk,
                "chunk_index": i
            })
        
        return chunk_data
    
    def create_collection(self, code_files: List[str], file_contents: Dict[str, str]):
        logger.info(f"Creating RAG collection with {len(code_files)} files")
        
        points = []
        point_id = 0
        
        for file_path in code_files:
            if file_path not in file_contents:
                continue
                
            content = file_contents[file_path]
            chunks = self._chunk_file_content(file_path, content)
            
            for chunk_data in chunks:
                embedding = self._get_embedding(chunk_data["content"])
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "file_path": chunk_data["file_path"],
                        "content": chunk_data["content"],
                        "chunk_id": chunk_data["chunk_id"],
                        "chunk_index": chunk_data["chunk_index"]
                    }
                )
                points.append(point)
                point_id += 1
        
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Indexed {len(points)} chunks in collection: {self.collection_name}")
    
    def search_relevant_files(self, bug_report: str, top_k: int = 20) -> List[str]:
        query_embedding = self._get_embedding(bug_report)
        
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        relevant_files = []
        seen_files = set()
        
        for result in search_results:
            file_path = result.payload["file_path"]
            if file_path not in seen_files:
                relevant_files.append(file_path)
                seen_files.add(file_path)
        
        logger.info(f"Found {len(relevant_files)} relevant files from RAG search: {self.collection_name}")
        return relevant_files
