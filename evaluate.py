"""Evaluate retrieval (ablation) and hallucination rate.

Dataset: eval/qa.jsonl, one JSON per line:
{"question": "...", "relevant_sources": ["case1.pdf"], "answerable": true}
(For unanswerable questions set "answerable": false and "relevant_sources": [].)

Usage: python evaluate.py --qa eval/qa.jsonl
"""
import argparse
import json

import numpy as np

from config import CFG
from pipeline import LegalQA


def load_qa(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def retrieval_metrics(qa: LegalQA, data, k=5):
    rows = [d for d in data if d.get("answerable", True)]
    print(f"\n== Retrieval ablation (n={len(rows)}, k={k}; relevance = doc-level match) ==")
    print(f"{'mode':<15}{'Recall@k':>10}{'MRR':>8}{'nDCG@k':>9}")
    res = {}
    for mode in qa.retriever.MODES:
        recalls, rr, ndcg = [], [], []
        for d in rows:
            rel = set(d["relevant_sources"])
            docs = qa.retriever.retrieve(d["question"], mode=mode, k=k)
            hits = [x.metadata["source"] in rel for x in docs]
            recalls.append(float(any(hits)))
            rr.append(next((1 / (i + 1) for i, h in enumerate(hits) if h), 0.0))
            dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
            idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k))) or 1
            ndcg.append(dcg / idcg)
        res[mode] = (np.mean(recalls), np.mean(rr), np.mean(ndcg))
        print(f"{mode:<15}{res[mode][0]:>10.3f}{res[mode][1]:>8.3f}{res[mode][2]:>9.3f}")
    base, best = res["dense"][2], res["hybrid_rerank"][2]
    print(f"\nRelative nDCG gain of hybrid+rerank over dense baseline: {100 * (best - base) / max(base, 1e-9):.1f}%")


def hallucination_metrics(qa: LegalQA, data):
    print(f"\n== Hallucination (n={len(data)}) ==")
    raw_h = final_h = 0
    for d in data:
        out = qa.ask(d["question"])
        answerable = d.get("answerable", True)
        # raw (pre-verification) hallucination: any unsupported sentence, or answered an unanswerable question
        raw_bad = out["unsupported_ratio"] > 0 or (not answerable and out["raw_answer"] and "could not find" not in out["raw_answer"].lower())
        # final: answered an unanswerable question (verified sentences are entailed by construction)
        final_bad = (not answerable) and not out["abstained"]
        raw_h += int(raw_bad)
        final_h += int(final_bad)
    n = len(data)
    print(f"Raw LLM hallucination rate : {100 * raw_h / n:.1f}%")
    print(f"After grounding filter     : {100 * final_h / n:.1f}%")
    print("Note: NLI-based proxy. Verify on a manually annotated subset before reporting numbers.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="eval/qa.jsonl")
    ap.add_argument("--k", type=int, default=CFG.final_k)
    a = ap.parse_args()
    data = load_qa(a.qa)
    qa = LegalQA()
    retrieval_metrics(qa, data, a.k)
    hallucination_metrics(qa, data)
