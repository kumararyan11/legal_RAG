import streamlit as st

from pipeline import LegalQA

st.set_page_config(page_title="Legal Document Q&A", page_icon="⚖️", layout="wide")
st.title("⚖️ Contextual Legal Q&A")
st.caption("Hybrid retrieval (BM25 + dense) → cross-encoder re-ranking → grounded, cited answers. Not legal advice.")


@st.cache_resource(show_spinner="Loading models and index...")
def get_qa():
    return LegalQA()


qa = get_qa()
q = st.text_input("Ask a question about your legal documents")

if q:
    with st.spinner("Searching and verifying..."):
        out = qa.ask(q)
    if out["abstained"]:
        st.warning(out["answer"])
    else:
        st.subheader("Answer")
        st.write(out["answer"])
        st.subheader("Sources")
        for c in out["citations"]:
            with st.expander(f"[{c['ref']}] {c['source']} — page {c['page']} (score {c['rerank_score']:.2f})"):
                st.write(c["passage"])
    if out["removed_sentences"]:
        with st.expander(f"Removed {len(out['removed_sentences'])} unsupported sentence(s)"):
            for s in out["removed_sentences"]:
                st.write("• " + s)
