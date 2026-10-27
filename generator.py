"""LLM wrapper: grounded, citation-forcing answer generation."""
from langchain_core.documents import Document

from config import CFG

SYSTEM_PROMPT = (
    "You are a careful legal research assistant. Answer the question using ONLY the numbered context passages.\n"
    "Rules:\n"
    "1. Every sentence of your answer must end with one or more citations like [1] or [2][3].\n"
    "2. Do not use outside knowledge. Do not speculate.\n"
    "3. If the passages do not contain the answer, reply exactly: "
    "\"I could not find this in the provided documents.\"\n"
    "4. Be concise and precise. This is research assistance, not legal advice."
)

ABSTAIN_TEXT = "I could not find this in the provided documents."


def build_context(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] (source: {d.metadata.get('source')}, page: {d.metadata.get('page', 0)})\n{d.page_content}")
    return "\n\n".join(parts)


class LLM:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        if cfg.llm_provider == "openai":
            from langchain_openai import ChatOpenAI
            self.client = ChatOpenAI(model=cfg.openai_model, temperature=0, max_tokens=cfg.max_new_tokens)
        else:
            from transformers import pipeline
            self.client = pipeline("text-generation", model=cfg.hf_model, device_map="auto", torch_dtype="auto")

    def generate(self, question: str, docs: list[Document]) -> str:
        user = f"Context passages:\n\n{build_context(docs)}\n\nQuestion: {question}\n\nAnswer:"
        if self.cfg.llm_provider == "openai":
            msgs = [("system", SYSTEM_PROMPT), ("human", user)]
            return self.client.invoke(msgs).content.strip()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        out = self.client(msgs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        return out[0]["generated_text"][-1]["content"].strip()
