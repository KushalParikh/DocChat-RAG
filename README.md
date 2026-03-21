# 🤖 DocChat: RAG System with Gemini & Streamlit

**Chat with your Data** – A modern Retrieval-Augmented Generation (RAG) system that allows users to upload documents (PDF, DOCX, TXT, CSV) and ask questions about them. The system uses **Google Gemini 2.5 Flash** for reasoning and **ChromaDB** for efficient vector storage.

## 🚀 Features
- **📄 Multi-Format Support**: Upload PDF, DOCX, TXT, and CSV files.
- **🧠 Local Embeddings**: Uses `sentence-transformers` for free, unlimited, and private vectorization.
- **⚡ High-Speed Inference**: Powered by Google's latest `gemini-2.5-flash` model.
- **💬 Conversational Memory**: Remembers context within the chat session.
- **🌊 Streaming Responses**: Real-time typewriter effect for answers.
- **📚 Sources Cited**: Shows the exact document snippets used for the answer.
- **🛠️ Utility Tools**: 
    - **Clear Database**: Reset the memory instantly.
    - **Download History**: Save your conversation as a text file.
- **🛡️ Cost-Efficient**: Fully optimized for the Free Tier (No Credit Card required).

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM**: Google Gemini 2.5 Flash (via `google-generativeai`)
- **Vector DB**: ChromaDB (Local)
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)
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

3.  **Setup API Key**
    - Get your free key from [Google AI Studio](https://aistudio.google.com/).
    - Create a `.env` file in the root folder:
      ```
      GOOGLE_API_KEY=your_key_here
      ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

> [!NOTE]
> **First Run Notice**: When you upload a document for the first time, the system will automatically download the embedding model (~90MB) to your machine. This helps keep your data private and avoids embedding API costs. Subsequent runs will be instant!


## 🛡️ Security Note
*   **Data Privacy**: The documents you upload are processed in memory and stored in a temporary local database. They are not sent to any third-party storage other than the Google Gemini API for processing.

## ⚠️ Limitations
*   **Single User Scope**: This project is designed as a portfolio demonstration for a single user. It uses a shared local vector store, meaning that in a concurrent multi-user deployment, users would share the same document context. For a production environment, this would be upgraded to use session-isolated databases (e.g., `chroma_db_{session_id}`).

## 📊 Evaluation
Uses [Ragas](https://docs.ragas.io/) to measure RAG pipeline quality across four industry-standard metrics:

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
- `app.py`: Main application logic (UI + RAG Pipeline).
- `evaluate.py`: Ragas evaluation script for RAG quality metrics.
- `requirements.txt`: Python dependencies.
- `.env`: Configuration for API keys (Hidden).
- `chroma_db/`: Local vector storage folder (Hidden/Temporary).
