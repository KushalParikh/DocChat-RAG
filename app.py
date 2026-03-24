import streamlit as st
import os
import uuid
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# DocChat modules
from cache import SemanticCache
from monitoring import QueryMonitor
from ingestion import process_documents
from retriever import build_retriever, classify_query_complexity

st.set_page_config(page_title="DocChat: Talk to your Data", page_icon="🤖")

st.title("🤖 DocChat")
st.caption("🚀 Chat with your documents using Google Gemini (Free & Fast)")


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialize all session state variables on first load."""
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())[:8]

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Upload a document and ask me anything about it!"}
        ]

    if "doc_hashes" not in st.session_state:
        st.session_state["doc_hashes"] = {}

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 0

    # Semantic cache and monitoring are stored as session state
    if "semantic_cache" not in st.session_state:
        st.session_state["semantic_cache"] = None  # Initialized after embedding model loads

    if "query_monitor" not in st.session_state:
        st.session_state["query_monitor"] = QueryMonitor()


init_session_state()

session_id = st.session_state["session_id"]
CHROMA_PATH = f"./chroma_db/{session_id}"


# ---------------------------------------------------------------------------
# Embedding Model (cached across sessions)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_embedding_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_semantic_cache():
    """Get or create the semantic cache (needs embedding model)."""
    if st.session_state["semantic_cache"] is None:
        st.session_state["semantic_cache"] = SemanticCache(
            embeddings=get_embedding_model(),
            threshold=0.92
        )
    return st.session_state["semantic_cache"]


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------

def get_vector_store(documents):
    """Create/update ChromaDB vector store from Document objects."""
    from langchain_community.vectorstores import Chroma

    embeddings = get_embedding_model()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    return vector_store


# ---------------------------------------------------------------------------
# Query Handler (with cache + adaptive retrieval + compression)
# ---------------------------------------------------------------------------

def user_input(user_question, chat_history):
    """
    Full query pipeline:
    1. Check semantic cache → instant return if hit
    2. Adaptive retrieval + reranking + compression
    3. Gemini LLM call
    4. Store result in cache
    """
    monitor = st.session_state["query_monitor"]
    cache = get_semantic_cache()
    start_time = time.time()

    # --- Step 1: Cache Check ---
    cache_result = cache.get(user_question)
    if cache_result is not None:
        answer, sources = cache_result
        latency = (time.time() - start_time) * 1000
        monitor.log_query(
            query=user_question, cache_hit=True,
            chunks_used=0, response_length=len(answer),
            k_used=0, latency_ms=latency
        )
        return answer, sources, True  # True = cache hit

    # --- Step 2: Retrieval ---
    embeddings = get_embedding_model()

    try:
        from langchain_community.vectorstores import Chroma
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

        # Build adaptive retriever with reranking
        from retriever import build_retriever
        retriever, k_used = build_retriever(vectorstore, embeddings, user_question)
        docs = retriever.invoke(user_question)

        context_text = "\n\n".join([doc.page_content for doc in docs])

        # --- Step 3: LLM Call ---
        llm_choice = st.session_state.get("llm_choice", "Groq (Llama-3.3-70B)")
        full_response = ""
        
        system_prompt = f"""You are DocChat, an intelligent document assistant. Answer the user's question based strictly on the provided Context.
Instructions:
1. Read the Context carefully.
2. If the answer is not in the Context, say "I cannot find the answer in the document."
3. Answer concisely in 3 sentences maximum unless requested otherwise.
4. Answer ONLY using the provided context. Do not use outside knowledge.
5. ALWAYS append the string "[SOURCE]" to the end of your answer if you used the Context.

Context:
{context_text}
"""
        
        human_prompt = f"Chat History:\n{chat_history}\n\nCurrent Question: {user_question}\nAnswer:"
        
        if "Groq" in llm_choice:
            from langchain_groq import ChatGroq
            from langchain_core.messages import SystemMessage, HumanMessage
            
            # Using Llama 3.3 70B via Groq for high speed and generous limits
            model = ChatGroq(
                model_name="llama-3.3-70b-versatile", 
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            response = model.invoke(messages)
            full_response = response.content
            
        else:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            
            combined_prompt = f"{system_prompt}\n\n{human_prompt}"
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content(combined_prompt)
            full_response = response.text

        latency = (time.time() - start_time) * 1000

        # --- Step 4: Cache the result ---
        source_texts = [doc.page_content for doc in docs]
        cache.set(user_question, full_response, source_texts)

        monitor.log_query(
            query=user_question, cache_hit=False,
            chunks_used=len(docs), response_length=len(full_response),
            k_used=k_used, latency_ms=latency
        )

        return full_response, docs, False  # False = cache miss

    except Exception as e:
        print(f"Query Error: {e}")
        return None, [], False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Core Settings")
    llm_choice = st.selectbox(
        "Choose AI Model",
        options=["Groq (Llama-3.3-70B)", "Google Gemini (2.5 Flash)"],
        index=0  # Groq is default
    )
    st.session_state["llm_choice"] = llm_choice
    
    st.divider()

    st.header("Upload Document")

    uploaded_files = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
        key=st.session_state["uploader_key"]
    )

    if st.button("Submit & Process"):
        with st.spinner("🔄 Processing documents with semantic chunking..."):
            try:
                embeddings = get_embedding_model()
                docs, updated_hashes, processed, skipped = process_documents(
                    uploaded_files,
                    st.session_state["doc_hashes"],
                    embeddings
                )

                st.session_state["doc_hashes"] = updated_hashes

                if skipped:
                    for name in skipped:
                        st.info(f"📋 **{name}** is already in the knowledge base — skipped.")

                if docs:
                    get_vector_store(docs)
                    # Invalidate cache since new docs may change answers
                    get_semantic_cache().invalidate()
                    chunk_count = len(docs)
                    st.success(f"✅ Processed {len(processed)} file(s) → {chunk_count} semantic chunks indexed!")
                elif not skipped:
                    st.warning("⚠️ No text found in the uploaded files.")

            except Exception as e:
                st.error("❌ Error processing files. Ensure they are valid PDF, DOCX, TXT, or CSV.")
                st.info(f"Technical Details: {str(e)}")

    st.divider()

    # Reset Button
    if st.button("🗑️ Clear Database / Reset"):
        try:
            from langchain_community.vectorstores import Chroma
            import shutil

            db_to_clear = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_model())
            try:
                db_to_clear.delete_collection()
            except Exception:
                pass

            if os.path.exists(CHROMA_PATH):
                try:
                    shutil.rmtree(CHROMA_PATH)
                except OSError:
                    pass

            # Reset all session state
            st.session_state["messages"] = [
                {"role": "assistant", "content": "Upload a document and ask me anything about it!"}
            ]
            st.session_state["doc_hashes"] = {}
            st.session_state["semantic_cache"] = None
            st.session_state["query_monitor"] = QueryMonitor()
            st.session_state["session_id"] = str(uuid.uuid4())[:8]
            st.session_state["uploader_key"] += 1
            st.cache_resource.clear()
            st.success("🗑️ Database cleared! Session reset.")
            st.rerun()
        except Exception as e:
            st.error(f"Error resetting: {e}")

    # Download Chat History
    chat_log = ""
    if "messages" in st.session_state:
        for msg in st.session_state["messages"]:
            chat_log += f"{msg['role'].upper()}: {msg['content']}\n\n"

    st.download_button(
        label="📥 Download Chat History",
        data=chat_log,
        file_name="chat_history.txt",
        mime="text/plain"
    )

    # ---------------------------------------------------------------------------
    # Admin Dashboard
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("📊 Session Dashboard")

    # Session info
    st.caption(f"🔑 Session: `{session_id}`")

    # Cache stats
    cache_stats = get_semantic_cache().stats()
    monitor_stats = st.session_state["query_monitor"].get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", monitor_stats["total_queries"])
        st.metric("Cache Hits", cache_stats["hits"])
    with col2:
        st.metric("Hit Rate", cache_stats["hit_rate"])
        st.metric("Est. Cost", monitor_stats["estimated_cost"])

    if monitor_stats["total_queries"] > 0:
        st.caption(f"📦 Avg chunks/query: {monitor_stats['avg_chunks_per_query']}")
        st.caption(f"⚡ Avg latency: {monitor_stats['avg_latency_ms']}ms")

    # Document registry
    doc_count = len(st.session_state["doc_hashes"])
    if doc_count > 0:
        st.caption(f"📄 Documents indexed: {doc_count}")
        with st.expander("View Documents"):
            for h, meta in st.session_state["doc_hashes"].items():
                st.text(f"• {meta['filename']} ({meta['chunk_count']} chunks)")


# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    st.chat_message(msg["role"], avatar=avatar).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="👤").write(prompt)

    if prompt:
        with st.chat_message("assistant", avatar="🤖"):
            try:
                # Format Chat History
                chat_history = ""
                for msg in st.session_state.messages[-10:]:  # Last 10 messages for context window
                    chat_history += f"{msg['role'].title()}: {msg['content']}\n"

                response_text, docs, cache_hit = user_input(prompt, chat_history)

                if response_text:
                    # Show cache indicator
                    if cache_hit:
                        st.caption("⚡ **Cache HIT** — instant response")
                    else:
                        k = classify_query_complexity(prompt)
                        st.caption(f"🔍 Retrieved with k={k} → {len(docs)} chunks after reranking")

                    st.write(response_text)

                    # Source Documents
                    show_sources = "[SOURCE]" in response_text
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                    if show_sources and docs and not cache_hit:
                        with st.expander("📚 Sources Cited"):
                            for i, doc in enumerate(docs):
                                st.write(f"**Source {i+1}**")
                                if hasattr(doc, 'metadata') and doc.metadata:
                                    meta_str = " | ".join(f"{k}: {v}" for k, v in doc.metadata.items()
                                                          if k not in ["doc_hash"])
                                    if meta_str:
                                        st.caption(f"📎 {meta_str}")
                                st.caption(doc.page_content[:500])
                                st.divider()

                    elif show_sources and cache_hit:
                        st.caption("📚 Sources from cached response")
                else:
                    st.error("Failed to generate response.")

            except Exception as e:
                st.error("⚠️ Oops! Something went wrong while generating the answer.")
                st.info("Tip: Try asking the question again or re-uploading your document.")
                print(f"Chat Error: {e}")
