"""Context grounding + citation tracing.

Each answer sentence must (a) carry a valid [n] citation and (b) be entailed by the cited passage(s)
according to an NLI model. Unsupported sentences are removed -> lower hallucination rate.
"""
import re
from dataclasses import dataclass, field

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from config import CFG
from generator import ABSTAIN_TEXT

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[\"'])")
CITE = re.compile(r"\[(\d+)\]")
ENTAIL_IDX = 1  # cross-encoder/nli-deberta-v3-*: [contradiction, entailment, neutral]


@dataclass
class SentenceCheck:
    text: str
    citations: list[int]
    entailment: float
    supported: bool
    reason: str = ""


@dataclass
class GroundingReport:
    sentences: list[SentenceCheck] = field(default_factory=list)
    abstained: bool = False

    @property
    def unsupported_ratio(self) -> float:
        if not self.sentences:
            return 0.0
        return sum(not s.supported for s in self.sentences) / len(self.sentences)

    @property
    def has_hallucination(self) -> bool:
        return any(not s.supported for s in self.sentences)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class GroundingVerifier:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.nli = CrossEncoder(cfg.nli_model)

    def verify(self, answer: str, docs: list[Document]) -> GroundingReport:
        report = GroundingReport()
        if ABSTAIN_TEXT.lower() in answer.lower():
            report.abstained = True
            return report

        for sent in [s.strip() for s in SENT_SPLIT.split(answer) if s.strip()]:
            cites = [int(c) for c in CITE.findall(sent)]
            valid = sorted({c for c in cites if 1 <= c <= len(docs)})
            claim = CITE.sub("", sent).strip()
            if not claim:
                continue
            if not valid:
                report.sentences.append(SentenceCheck(sent, cites, 0.0, False, "missing/invalid citation"))
                continue
            logits = np.array(self.nli.predict([(docs[c - 1].page_content, claim) for c in valid]))
            probs = _softmax(logits.reshape(len(valid), -1))
            best = float(probs[:, ENTAIL_IDX].max())
            ok = best >= self.cfg.entail_threshold
            report.sentences.append(SentenceCheck(sent, valid, best, ok, "" if ok else "not entailed by cited passage"))
        return report
