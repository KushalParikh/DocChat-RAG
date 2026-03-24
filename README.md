# 🤖 DocChat: RAG System with Gemini & Streamlit

**Chat with your Data** – A modern Retrieval-Augmented Generation (RAG) system that allows users to upload documents (PDF, DOCX, TXT, CSV) and ask questions about them. The system uses **Google Gemini 2.5 Flash** for reasoning and **ChromaDB** for efficient vector storage.

## 🚀 Features
- **📄 Multi-Format Support**: Upload PDF, DOCX, TXT, and CSV files.
- **⚡ Semantic Caching**: Instant, zero-cost responses for repeated or similarly phrased questions. (NEW)
- **🧠 Advanced Chunking**: Uses `SemanticChunker` to split documents by natural topic boundaries instead of arbitrary character counts. (NEW)
- **🎯 Adaptive Retrieval & Reranking**: Dynamically adjusts chunks retrieved based on query complexity and uses a Cross-Encoder to re-score relevance. (NEW)
- **🗜️ Context Compression**: Filters out irrelevant sentences within chunks before sending to the LLM to save tokens. (NEW)
- **🛡️ Session-Isolated Data**: Vector stores are scoped to the active session (`chroma_db/<session_id>`) for multi-user safety. (NEW)
- **📊 Admin Dashboard**: Real-time tracking of cache hit rates, chunks used, and estimated API costs. (NEW)
- **💡 High-Speed Inference**: Choose between Google's `gemini-2.5-flash` or Groq's high-limit `Llama-3.3-70B` model (Default).
- **📚 Sources Cited**: Shows the exact document snippets used for the answer.

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM**: Groq (Llama-3.3-70B) or Google Gemini 2.5 Flash
- **Vector DB**: ChromaDB (Local)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
- **Reranker**: HuggingFace Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
- **Orchestration**: LangChain

## 🏃‍♂️ How to Run (Local)

1.  **Clone the repository**
    ```bash
    git clone https://github.com/KushalParikh/DocChat-RAG.git
    cd DocChat-RAG
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup API Keys**
    - Get keys from [Groq Console](https://console.groq.com/keys) and [Google AI Studio](https://aistudio.google.com/).
    - Create a `.env` file in the root folder:
      ```
      GROQ_API_KEY=your_groq_key_here
      GOOGLE_API_KEY=your_gemini_key_here
      ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

> [!NOTE]
> **First Run Notice**: When you upload a document for the first time, the system will automatically download the embedding and reranker models to your machine. Subsequent runs will be instant!


## 🛡️ Security Note
*   **Data Privacy**: The documents you upload are processed in memory and stored in a temporary local database scoped to your session. They are not sent to any third-party storage other than the Google Gemini API for processing.

## 📊 Evaluation
Uses [Ragas](https://docs.ragas.io/) to measure RAG pipeline quality across industry-standard metrics, plus tracks pipeline efficiency:

| Metric | What it Measures |
|---|---|
| **Faithfulness** | Does the answer only use information from retrieved chunks? |
| **Answer Relevancy** | Does the answer actually address the question asked? |
| **Context Precision** | Was the retrieved context relevant to the question? |
| **Context Recall** | Did retrieval find all the relevant chunks? |

**How to run:**
1. Process at least one document in the app first.
2. Update the test questions in `evaluate.py` to match your document.
3. Run:
    ```bash
    python evaluate.py
    ```
Results are printed to the console and saved to `evaluation_results.csv`.

## 📂 Project Structure
- `app.py`: Main application UI and session management.
- `ingestion.py`: Document deduplication and semantic chunking.
- `retriever.py`: Adaptive retrieval, reranking, and context compression.
- `cache.py`: Semantic cache for optimizing duplicate queries.
- `monitoring.py`: Query logging and Gemini cost estimation.
- `evaluate.py`: Evaluation script for tracking pipeline metrics.
- `requirements.txt`: Python dependencies.
- `chroma_db/`: Local vector storage folder (session-scoped).
