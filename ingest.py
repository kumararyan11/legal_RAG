"""Build the FAISS (dense) index and persist chunks for BM25.

Usage: python ingest.py            # reads PDFs/TXTs from data/docs
"""
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config import CFG


def load_documents(data_dir: Path) -> list[Document]:
    docs: list[Document] = []
    files = [p for p in sorted(data_dir.rglob("*")) if p.suffix.lower() in {".pdf", ".txt"}]
    for path in tqdm(files, desc="Loading"):
        try:
            loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
            for d in loader.load():
                d.metadata["source"] = path.name
                d.metadata.setdefault("page", 0)
                docs.append(d)
        except Exception as e:  # skip corrupt files but keep going
            print(f"[skip] {path.name}: {e}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG.chunk_size,
        chunk_overlap=CFG.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def main():
    docs = load_documents(CFG.data_dir)
    if not docs:
        raise SystemExit(f"No .pdf/.txt files found in {CFG.data_dir}")
    chunks = chunk_documents(docs)
    print(f"{len(docs)} pages/docs -> {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=CFG.embedding_model,
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
    vs = FAISS.from_documents(chunks, embeddings)

    CFG.index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(CFG.index_dir))
    with open(CFG.index_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index saved to {CFG.index_dir}")


if __name__ == "__main__":
    main()
