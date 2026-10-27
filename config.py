import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent


@dataclass
class Config:
    # paths
    data_dir: Path = ROOT / "data" / "docs"
    index_dir: Path = ROOT / "index"

    # chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150

    # models
    embedding_model: str = "BAAI/bge-small-en-v1.5"           # open-source dense embeddings
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-small"      # labels: contradiction, entailment, neutral

    # retrieval
    top_k_dense: int = 25
    top_k_bm25: int = 25
    rrf_k: int = 60                  # reciprocal-rank-fusion constant
    rerank_candidates: int = 30      # fused candidates sent to cross-encoder
    final_k: int = 5                 # chunks given to the LLM
    min_rerank_score: float = -4.0   # abstain below this (tune on dev set)

    # generation
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")   # "openai" | "hf"
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    hf_model: str = os.getenv("HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    max_new_tokens: int = 400

    # grounding
    entail_threshold: float = 0.50   # min entailment prob for a sentence to count as supported


CFG = Config()
