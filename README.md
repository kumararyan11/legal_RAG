# Contextual Q&A System for Legal Document Search (RAG)

Pipeline: PDFs/TXT -> chunking -> FAISS (dense) + BM25 (sparse) -> RRF fusion -> cross-encoder re-rank
-> LLM answer with [n] citations -> NLI sentence-level grounding check -> cited answer or abstain.

## Setup
    pip install -r requirements.txt
    export OPENAI_API_KEY=...            # or: export LLM_PROVIDER=hf  (fully open-source, local)

## Run
    # 1. put legal PDFs/TXTs in data/docs/
    python ingest.py
    # 2. CLI
    python pipeline.py "What is the limitation period for recovery of money?"
    # 3. UI
    streamlit run app.py
    # 4. Evaluation (write your own eval/qa.jsonl)
    python evaluate.py --qa eval/qa.jsonl

## Datasets (for the 10,000+ docs)
Indian Supreme Court judgments (AWS Open Data), ILDC, CUAD contracts, Pile of Law.

## Tuning
config.py: chunk_size, top_k_*, final_k, min_rerank_score (abstention), entail_threshold (grounding strictness).
