# backend.py
import os
import asyncio
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from neo4j import AsyncGraphDatabase

load_dotenv()

class HybridRAGEngine:
    def __init__(self, openai_key: str, neo4j_url: str, neo4j_user: str, neo4j_pass: str, faiss_path: str = "faiss_index"):
        self.openai_key = openai_key
        self.neo4j_url = neo4j_url
        self.neo4j_user = neo4j_user
        self.neo4j_pass = neo4j_pass
        self.faiss_path = faiss_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    def verify_vector_db_exists(self) -> bool:
        """Verifies if the index configuration path exists on local disk storage."""
        return os.path.exists(os.path.join(self.faiss_path, "index.faiss"))

    async def _fetch_vector_context(self, query: str) -> str:
        """Asynchronously triggers the blocking local FAISS IO lookup via a worker thread pool."""
        loop = asyncio.get_running_loop()
        
        def sync_vector_search():
            vectorstore = FAISS.load_local(self.faiss_path, self.embeddings, allow_dangerous_deserialization=True)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            return retriever.get_relevant_documents(query)

        matched_docs = await loop.run_in_executor(None, sync_vector_search)
        return "\n".join([doc.page_content for doc in matched_docs])

    async def _fetch_graph_context(self, query: str):
        """Asynchronously executes a non-blocking 2-hop deep relational graph discovery query."""
        cypher_search = """
        MATCH (s:Entity)-[r1]->(o:Entity)
        WHERE toLower(s.name) CONTAINS toLower($query) OR toLower(o.name) CONTAINS toLower($query)
        OPTIONAL MATCH (o)-[r2]->(next:Entity)
        RETURN s.name AS subject, type(r1) AS predicate, o.name AS object,
               next.name AS next_target, type(r2) AS next_predicate
        LIMIT 20
        """
        graph_context_list = []
        raw_triples = []
        
        async with AsyncGraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_user, self.neo4j_pass)) as driver:
            async with driver.session() as session:
                result = await session.run(cypher_search, query=query)
                async for record in result:
                    sub, pred, obj = record['subject'], record['predicate'], record['object']
                    graph_context_list.append(f"({sub})-[{pred}]->({obj})")
                    raw_triples.append({"source": sub, "type": pred, "target": obj})
                    
                    if record['next_target'] and record['next_predicate']:
                        n_pred, n_targ = record['next_predicate'], record['next_target']
                        graph_context_list.append(f"({obj})-[{n_pred}]->({n_targ})")
                        raw_triples.append({"source": obj, "type": n_pred, "target": n_targ})
                        
        unique_context = "\n".join(list(set(graph_context_list)))
        
        # Deduplicate the structural map elements for clear rendering parameters
        seen = set()
        deduped_triples = []
        for t in raw_triples:
            identifier = (t["source"], t["type"], t["target"])
            if identifier not in seen:
                seen.add(identifier)
                deduped_triples.append(t)
                
        return unique_context, deduped_triples

    async def execute_hybrid_query_async(self, query: str):
        """Orchestrates high-speed concurrent search pathways across vector and graph systems."""
        if not self.verify_vector_db_exists():
            raise FileNotFoundError("FAISS vector database path not detected on disk storage.")

        vector_task = self._fetch_vector_context(query)
        graph_task = self._fetch_graph_context(query)
        
        vector_context, (graph_context, raw_triples) = await asyncio.gather(vector_task, graph_task)
        
        synthesis_llm = ChatOpenAI(temperature=0.2, model="gpt-4o-mini", openai_api_key=self.openai_key)
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an advanced Hybrid RAG system. Answer the query by accurately blending structured facts from a Knowledge Graph and raw details from Text Chunks.\n\n"
                "=== TEXT CHUNK CONTEXT ===\n{vector_context}\n\n"
                "=== KNOWLEDGE GRAPH RELATIONSHIPS ===\n{graph_context}"
            )),
            ("human", "{question}")
        ])
        
        synthesis_chain = synthesis_prompt | synthesis_llm
        response = await synthesis_chain.ainvoke({
            "vector_context": vector_context,
            "graph_context": graph_context,
            "question": query
        })
        
        return response.content, vector_context, graph_context, raw_triples
