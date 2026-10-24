# frontend/frontend.py
import os
import sys
import asyncio
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from streamlit_agraph import agraph, Node, Edge, Config

# 1. DYNAMIC SYSTEM PATH REGISTRATION (Ensures backend folder is found flawlessly)
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Now we can import comfortably without relative path conflicts
import backend.ingestion as ingestion
from backend.backend import HybridRAGEngine

load_dotenv()

st.set_page_config(page_title="Modular Hybrid Graph-RAG Engine", layout="wide")
st.title("🌐 Decoupled Neo4j & FAISS Hybrid RAG Application")

with st.sidebar:
    st.header("⚙️ Environment Status")
    has_openai = "✅ Configured" if os.getenv("OPENAI_API_KEY") else "❌ Missing"
    has_neo4j = "✅ Configured" if os.getenv("NEO4J_URI") and os.getenv("NEO4J_PASSWORD") else "❌ Missing"
    
    st.write(f"**OpenAI Connectivity:** {has_openai}")
    st.write(f"**Neo4j Connectivity:** {has_neo4j}")
    st.markdown("---")
    st.caption("Settings are loaded securely from the local `.env` configuration file.")

tab1, tab2 = st.tabs(["📁 Ingest Knowledge Domain", "🔍 Analytics Discovery Hub"])

with tab1:
    st.subheader("Ingest Knowledge Base Documents")
    uploaded_file = st.file_uploader("Upload reference documents for parsing:", type=["txt", "pdf"])
    
    if st.button("Process & Build Databases"):
        if not uploaded_file:
            st.error("Please provide a valid file context source before executing.")
        else:
            with st.spinner("Ingestion pipeline processing elements..."):
                try:
                    # --- THE FIX: RAW BYTES EXTRACTION ---
                    # We extract the pure file contents and filename string explicitly.
                    # This prevents Streamlit from passing wrapped object tuples to LangChain loaders.
                    file_name = uploaded_file.name
                    file_bytes = uploaded_file.getvalue()  # Read raw stream data directly
                    
                    # Pass the primitive variables explicitly to the ingestion layer
                    success = ingestion.run_pipeline(
                        file_name=file_name,
                        file_bytes=file_bytes
                    )
                    
                    if success:
                        st.success(f"Success! '{file_name}' has been processed. Data loaded successfully.")
                except Exception as e:
                    st.error(f"Execution error encountered: {str(e)}")

with tab2:
    st.subheader("Execute Multi-Channel Semantic Retrieval")
    user_query = st.text_input("Ask a question about concepts inside your knowledge network:")
    
    if st.button("Search Pipeline"):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OpenAI verification parameters missing from environment runtime context.")
        elif not user_query.strip():
            st.warning("Please type a valid lookup query.")
        else:
            engine = HybridRAGEngine(
                openai_key=os.getenv("OPENAI_API_KEY"),
                neo4j_url=os.getenv("NEO4J_URI"),
                neo4j_user=os.getenv("NEO4J_USERNAME"),
                neo4j_pass=os.getenv("NEO4J_PASSWORD")
            )
            
            if not engine.verify_vector_db_exists():
                st.error("No active FAISS context databases discovered. Ingest data in Tab 1 first.")
            else:
                with st.spinner("Retrieving concurrently from FAISS and Neo4j..."):
                    try:
                        answer, vec_ctx, graph_ctx, raw_triples = asyncio.run(
                            engine.execute_hybrid_query_async(user_query)
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("💡 **FAISS Vector Chunks Retrieved:**")
                            st.caption(vec_ctx if vec_ctx else "No direct textual chunks found.")
                        with col2:
                            st.markdown("🕸️ **Neo4j Cypher Path Entities:**")
                            st.caption(graph_ctx if graph_ctx else "No overlapping networks discovered.")
                        
                        st.markdown("### 🎯 Hybrid Synthesis Response")
                        st.write(answer)
                        
                        if raw_triples:
                            st.markdown("### 🗺️ Network Topology Exploration Map")
                            nodes = set()
                            edges = []
                            
                            for triple in raw_triples:
                                nodes.add(triple["source"])
                                nodes.add(triple["target"])
                                edges.append(Edge(source=triple["source"], target=triple["target"], label=triple["type"]))
                                
                            node_objects = [
                                Node(id=name, label=name, size=15, color="#4F46E5", font={'color': 'white'}) 
                                for name in nodes
                            ]
                            
                            config = Config(width=1000, height=450, directed=True, physics=True, hierarchical=False)
                            agraph(nodes=node_objects, edges=edges, config=config)
                            
                    except Exception as ex:
                        st.error(f"Error handling retrieval operations: {str(ex)}")
