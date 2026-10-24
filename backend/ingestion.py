# backend/ingestion.py
import os
import tempfile
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document

load_dotenv()

class RelationTriple(BaseModel):
    subject: str = Field(description="The source entity name (noun). Example: 'LangChain'")
    predicate: str = Field(description="The relationship verb in uppercase with underscores. Example: 'INTEGRATES_WITH'")
    object: str = Field(description="The target entity name (noun). Example: 'Neo4j'")

class KnowledgeGraph(BaseModel):
    triples: List[RelationTriple] = Field(description="List of unique entity relationships found in the text.")

def load_bytes_to_docs(file_name: str, file_bytes: bytes):
    """Saves raw bytes into a temporary path location and runs clean native loaders."""
    file_extension = os.path.splitext(file_name).lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return loader.load()
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def run_pipeline(file_name: str, file_bytes: bytes, faiss_path: str = "faiss_index"):
    """Accepts explicit primitives to securely build vector indexes and graph elements."""
    openai_key = os.getenv("OPENAI_API_KEY")
    neo4j_url = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_pass = os.getenv("NEO4J_PASSWORD")

    if not all([openai_key, neo4j_url, neo4j_user, neo4j_pass]):
        raise ValueError("Missing required configuration values in your environment or .env file.")

    # Load from safe raw bytes array
    docs = load_bytes_to_docs(file_name, file_bytes)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    raw_texts = [str(doc.page_content).strip() for doc in docs if hasattr(doc, 'page_content')]
    chunks = text_splitter.create_documents(raw_texts)
    
    if not chunks:
        raise ValueError("No text content could be extracted.")
    
    # Core Vector Persistence
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_path)
    
    # Knowledge Graph Extraction
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=openai_key)
    graph = Neo4jGraph(url=neo4j_url, username=neo4j_user, password=neo4j_pass)
    
    structured_llm = llm.with_structured_output(KnowledgeGraph)
    kg_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract key entities and relationships as subject-predicate-object triples from the text."),
        ("human", "Analyze this text segment:\n\n{text}")
    ])
    extraction_chain = kg_prompt | structured_llm
    
    for chunk in chunks:
        try:
            text_to_analyze = str(chunk.page_content).strip()
            if not text_to_analyze:
                continue
                
            extracted_output = extraction_chain.invoke({"text": text_to_analyze})
            
            triples_list = []
            if isinstance(extracted_output, dict):
                triples_list = extracted_output.get("triples", [])
            elif isinstance(extracted_output, KnowledgeGraph):
                triples_list = extracted_output.triples
            else:
                continue

            for raw_triple in triples_list:
                if isinstance(raw_triple, dict):
                    raw_sub = raw_triple.get("subject", "Unknown")
                    raw_pred = raw_triple.get("predicate", "RELATED_TO")
                    raw_obj = raw_triple.get("object", "Unknown")
                else:
                    raw_sub = getattr(raw_triple, "subject", "Unknown")
                    raw_pred = getattr(raw_triple, "predicate", "RELATED_TO")
                    raw_obj = getattr(raw_triple, "object", "Unknown")

                sub_str = str(raw_sub).strip().title()
                obj_str = str(raw_obj).strip().title()
                pred_string = str(raw_pred).strip().replace(' ', '_').upper()
                
                if not pred_string or pred_string == "TUPLE":
                    pred_string = "RELATED_TO"
                
                cypher_query = f"""
                MERGE (s:Entity {{name: $subject}})
                MERGE (o:Entity {{name: $object}})
                MERGE (s)-[r:{pred_string}]->(o)
                """
                graph.query(cypher_query, {"subject": sub_str, "object": obj_str})
        except Exception:
            continue
            
    return True
