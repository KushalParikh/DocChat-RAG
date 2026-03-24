"""
DocChat RAG Evaluation Script
=============================
Uses the Ragas framework to evaluate the RAG pipeline across four key metrics:
  - Faithfulness:       Does the answer only use information from retrieved chunks?
  - Answer Relevancy:   Does the answer actually address the question asked?
  - Context Precision:  Was the retrieved context relevant to the question?
  - Context Recall:     Did retrieval find all the relevant chunks?

Additional metrics tracked after upgrades:
  - Chunks per query:   Mean chunks passed to LLM (validates adaptive k)
  - Prompt size:        Average prompt token count (tracks compression effectiveness)

Usage:
  1. Make sure you have documents processed in chroma_db/ (run the app first).
  2. Update the TEST_DATASET below with questions relevant to YOUR documents.
  3. Run: python evaluate.py
"""

import os
import time
from dotenv import load_dotenv
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "models/gemini-2.5-flash"

# ---------------------------------------------------------------------------
# 2. Test Dataset  (Update these with questions relevant to YOUR documents)
# ---------------------------------------------------------------------------
TEST_DATASET = [
    {
        "question": "What is the main topic of the document?",
        "ground_truth": "UPDATE THIS — write the actual main topic of your uploaded document.",
    },
    {
        "question": "What are the key findings or conclusions?",
        "ground_truth": "UPDATE THIS — write the actual key findings from your document.",
    },
    {
        "question": "Who is the author of the document?",
        "ground_truth": "UPDATE THIS — write the actual author name.",
    },
    {
        "question": "What methodology or approach is described?",
        "ground_truth": "UPDATE THIS — write the actual methodology described.",
    },
    {
        "question": "What recommendations are made in the document?",
        "ground_truth": "UPDATE THIS — write the actual recommendations from your document.",
    },
]


def get_embedding_model():
    """Load the same embedding model used by the main app."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def run_rag_pipeline(question: str) -> dict:
    """
    Run a single question through the upgraded RAG pipeline.
    Uses the same retriever module as app.py for consistency.
    Returns the generated answer, retrieved context chunks, and pipeline stats.
    """
    from langchain_community.vectorstores import Chroma
    import google.generativeai as genai
    from retriever import build_retriever, classify_query_complexity

    genai.configure(api_key=GOOGLE_API_KEY)

    embeddings = get_embedding_model()

    # Find the first session directory in chroma_db
    chroma_path = CHROMA_DB_PATH
    if os.path.isdir(CHROMA_DB_PATH):
        subdirs = [d for d in os.listdir(CHROMA_DB_PATH)
                    if os.path.isdir(os.path.join(CHROMA_DB_PATH, d))]
        if subdirs:
            chroma_path = os.path.join(CHROMA_DB_PATH, subdirs[0])

    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)

    # Use the upgraded retriever pipeline
    start_time = time.time()
    retriever, k_used = build_retriever(db, embeddings, question)
    docs = retriever.invoke(question)
    retrieval_time = time.time() - start_time

    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n".join(contexts)

    # Generate answer with Gemini
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""Answer the question based strictly on the provided context.
If the answer is not in the context, say "I cannot find the answer in the document."
Answer concisely in 3 sentences maximum unless the question explicitly requires more.

Context:
{context_text}

Question:
{question}

Answer:"""

    gen_start = time.time()
    response = model.generate_content(prompt)
    gen_time = time.time() - gen_start
    answer = response.text

    # Estimate prompt size (rough: 4 chars ≈ 1 token)
    prompt_tokens = len(prompt) // 4

    return {
        "answer": answer,
        "contexts": contexts,
        "k_used": k_used,
        "chunks_after_rerank": len(docs),
        "prompt_tokens": prompt_tokens,
        "retrieval_time_ms": round(retrieval_time * 1000),
        "generation_time_ms": round(gen_time * 1000),
    }


def build_evaluation_dataset() -> tuple[Dataset, list[dict]]:
    """Run every test question through the RAG pipeline and build a HuggingFace Dataset."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    pipeline_stats = []

    print("Running RAG pipeline on test questions...\n")
    for i, item in enumerate(TEST_DATASET, 1):
        q = item["question"]
        print(f"  [{i}/{len(TEST_DATASET)}] {q}")

        result = run_rag_pipeline(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])
        pipeline_stats.append(result)

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    return dataset, pipeline_stats


def main():
    # Preflight checks
    if not GOOGLE_API_KEY:
        print("ERROR: GOOGLE_API_KEY not found. Add it to your .env file.")
        return

    if not os.path.exists(CHROMA_DB_PATH):
        print("ERROR: chroma_db/ not found. Upload & process a document in the app first.")
        return

    print("=" * 60)
    print("  DocChat — RAG Evaluation (Ragas)")
    print("=" * 60)

    # Build dataset by running the pipeline
    dataset, pipeline_stats = build_evaluation_dataset()

    # Configure Ragas to use Gemini via LangChain wrapper
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas.llms import LangchainLLMWrapper

    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY,
        )
    )

    # Run evaluation
    print("\nEvaluating with Ragas (this may take a minute)...\n")
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
    )

    # Display Ragas results
    print("=" * 60)
    print("  RAGAS RESULTS")
    print("=" * 60)
    print(f"  Faithfulness:        {result['faithfulness']:.4f}")
    print(f"  Answer Relevancy:    {result['answer_relevancy']:.4f}")
    print(f"  Context Precision:   {result['context_precision']:.4f}")
    print(f"  Context Recall:      {result['context_recall']:.4f}")
    print("=" * 60)

    # Display pipeline metrics
    print("\n" + "=" * 60)
    print("  PIPELINE METRICS (New)")
    print("=" * 60)

    avg_k = sum(s["k_used"] for s in pipeline_stats) / len(pipeline_stats)
    avg_chunks = sum(s["chunks_after_rerank"] for s in pipeline_stats) / len(pipeline_stats)
    avg_prompt = sum(s["prompt_tokens"] for s in pipeline_stats) / len(pipeline_stats)
    avg_retrieval = sum(s["retrieval_time_ms"] for s in pipeline_stats) / len(pipeline_stats)
    avg_gen = sum(s["generation_time_ms"] for s in pipeline_stats) / len(pipeline_stats)

    print(f"  Avg k requested:     {avg_k:.1f}")
    print(f"  Avg chunks to LLM:   {avg_chunks:.1f}")
    print(f"  Avg prompt tokens:   {avg_prompt:.0f}")
    print(f"  Avg retrieval time:  {avg_retrieval:.0f}ms")
    print(f"  Avg generation time: {avg_gen:.0f}ms")
    print("=" * 60)

    # Save detailed results to CSV
    df = result.to_pandas()
    # Add pipeline stats columns
    for key in ["k_used", "chunks_after_rerank", "prompt_tokens", "retrieval_time_ms", "generation_time_ms"]:
        df[key] = [s[key] for s in pipeline_stats]

    output_file = "evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
