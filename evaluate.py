"""
DocChat RAG Evaluation Script
=============================
Uses the Ragas framework to evaluate the RAG pipeline across four key metrics:
  - Faithfulness:       Does the answer only use information from retrieved chunks?
  - Answer Relevancy:   Does the answer actually address the question asked?
  - Context Precision:  Was the retrieved context relevant to the question?
  - Context Recall:     Did retrieval find all the relevant chunks?

Usage:
  1. Make sure you have documents processed in chroma_db/ (run the app first).
  2. Update the TEST_DATASET below with questions relevant to YOUR documents.
  3. Run: python evaluate.py
"""

import os
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
# Each entry needs:
#   - question:     The user query
#   - ground_truth: The ideal / expected answer (used for recall & precision)
#
# 'answer' and 'contexts' are generated automatically by running the RAG
# pipeline below, so you only need to provide question + ground_truth.
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
    Run a single question through the same RAG pipeline used by app.py.
    Returns the generated answer and retrieved context chunks.
    """
    from langchain_community.vectorstores import Chroma
    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)

    # Retrieve relevant chunks
    embeddings = get_embedding_model()
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    docs = db.similarity_search(question, k=4)
    contexts = [doc.page_content for doc in docs]
    context_text = "\n\n".join(contexts)

    # Generate answer with Gemini
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = f"""Answer the question based strictly on the provided context.
If the answer is not in the context, say "I cannot find the answer in the document."

Context:
{context_text}

Question:
{question}

Answer:"""

    response = model.generate_content(prompt)
    answer = response.text

    return {"answer": answer, "contexts": contexts}


def build_evaluation_dataset() -> Dataset:
    """Run every test question through the RAG pipeline and build a HuggingFace Dataset."""
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Running RAG pipeline on test questions...\n")
    for i, item in enumerate(TEST_DATASET, 1):
        q = item["question"]
        print(f"  [{i}/{len(TEST_DATASET)}] {q}")

        result = run_rag_pipeline(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


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
    dataset = build_evaluation_dataset()

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

    # Display results
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Faithfulness:        {result['faithfulness']:.4f}")
    print(f"  Answer Relevancy:    {result['answer_relevancy']:.4f}")
    print(f"  Context Precision:   {result['context_precision']:.4f}")
    print(f"  Context Recall:      {result['context_recall']:.4f}")
    print("=" * 60)

    # Save detailed results to CSV
    df = result.to_pandas()
    output_file = "evaluation_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
