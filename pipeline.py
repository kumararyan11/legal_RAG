"""End-to-end Legal QA: retrieve -> generate -> verify -> trace citations."""
from config import CFG
from generator import ABSTAIN_TEXT, LLM
from grounding import GroundingVerifier
from retriever import HybridRetriever


class LegalQA:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.retriever = HybridRetriever(cfg)
        self.llm = LLM(cfg)
        self.verifier = GroundingVerifier(cfg)

    def ask(self, question: str) -> dict:
        docs = self.retriever.retrieve(question, mode="hybrid_rerank", k=self.cfg.final_k)

        # Gate 1: retrieval confidence
        if not docs or docs[0].metadata["score"] < self.cfg.min_rerank_score:
            return self._result(ABSTAIN_TEXT, [], None, raw=None, abstained=True)

        raw = self.llm.generate(question, docs)

        # Gate 2: sentence-level grounding
        report = self.verifier.verify(raw, docs)
        if report.abstained:
            return self._result(ABSTAIN_TEXT, [], report, raw=raw, abstained=True)

        kept = [s for s in report.sentences if s.supported]
        if not kept:
            return self._result(ABSTAIN_TEXT, [], report, raw=raw, abstained=True)

        answer = " ".join(s.text for s in kept)
        used = sorted({c for s in kept for c in s.citations})
        citations = [
            {
                "ref": c,
                "source": docs[c - 1].metadata.get("source"),
                "page": docs[c - 1].metadata.get("page", 0),
                "chunk_id": docs[c - 1].metadata.get("chunk_id"),
                "rerank_score": docs[c - 1].metadata.get("score"),
                "passage": docs[c - 1].page_content,
            }
            for c in used
        ]
        return self._result(answer, citations, report, raw=raw, abstained=False)

    @staticmethod
    def _result(answer, citations, report, raw, abstained):
        return {
            "answer": answer,
            "citations": citations,
            "abstained": abstained,
            "raw_answer": raw,
            "removed_sentences": [s.text for s in report.sentences if not s.supported] if report else [],
            "unsupported_ratio": report.unsupported_ratio if report else 0.0,
        }


if __name__ == "__main__":
    import sys
    qa = LegalQA()
    out = qa.ask(" ".join(sys.argv[1:]) or "What is the limitation period for filing a suit?")
    print(out["answer"], "\n")
    for c in out["citations"]:
        print(f"[{c['ref']}] {c['source']} p.{c['page']}")
