# Contextual Legal Q&A System (RAG)

A retrieval-augmented generation (RAG) application for asking questions
about a collection of legal documents. It combines keyword and semantic
retrieval, re-ranks candidate passages, generates citation-based
answers, and checks each answer sentence against the cited source text
before returning it.

> **Disclaimer:** This project is a legal-research aid, not legal
> advice. Verify all outputs against the original documents and consult
> a qualified legal professional for legal decisions.

## Overview

Legal documents can be lengthy, and a language model may produce
plausible answers that are not supported by the source material. This
project addresses that problem with a retrieval and verification
pipeline designed to keep answers tied to the supplied documents.

### Main features

-   **PDF and TXT ingestion:** loads documents from `data/docs/` and
    splits them into overlapping text chunks.
-   **Dense retrieval:** uses Hugging Face embeddings and FAISS to find
    semantically related passages.
-   **Keyword retrieval:** uses BM25 to retrieve passages with relevant
    terms.
-   **Hybrid ranking:** combines dense and BM25 rankings with Reciprocal
    Rank Fusion (RRF).
-   **Cross-encoder re-ranking:** re-scores fused candidates for
    relevance to the question.
-   **Citation-based generation:** instructs the LLM to answer only from
    the retrieved passages and attach `[n]` citations.
-   **Sentence-level grounding check:** uses an NLI cross-encoder to
    check whether cited passages entail each answer sentence.
-   **Abstention:** returns a fixed "not found" response when retrieval
    confidence is too low or no answer sentence passes verification.
-   **Streamlit interface:** displays the answer, source passages, page
    metadata, and any unsupported sentences removed by the verifier.
-   **Retrieval evaluation:** compares dense, BM25, hybrid, and
    hybrid-plus-reranking modes with Recall@k, MRR, and nDCG@k.

## Architecture

``` text
PDF / TXT files
      |
      v
Document loading and chunking
      |
      +-----------------------------+
      |                             |
      v                             v
Dense embeddings + FAISS          BM25 keyword index
      |                             |
      +-------------+---------------+
                    |
                    v
         Reciprocal Rank Fusion
                    |
                    v
       Cross-encoder re-ranking
                    |
                    v
       Top-k passages + citations
                    |
                    v
       LLM answer generation
                    |
                    v
      Sentence-level NLI checking
                    |
          +---------+---------+
          |                   |
          v                   v
   Supported sentences     Abstain / remove
          |
          v
  Answer with source citations
```

## Technology stack

  ----------------------------------------------------------------------------
  Component                           Implementation
  ----------------------------------- ----------------------------------------
  Application UI                      Streamlit

  RAG orchestration and document      LangChain
  objects                             

  Document loaders                    `PyPDFLoader`, `TextLoader`

  Text chunking                       `RecursiveCharacterTextSplitter`

  Embeddings                          `BAAI/bge-small-en-v1.5`

  Dense vector search                 FAISS

  Keyword retrieval                   BM25 via `rank-bm25`

  Rank fusion                         Reciprocal Rank Fusion (RRF)

  Re-ranking                          `cross-encoder/ms-marco-MiniLM-L-6-v2`

  Grounding verification              `cross-encoder/nli-deberta-v3-small`

  LLM provider                        OpenAI API by default, or a local
                                      Hugging Face text-generation model

  Evaluation                          Recall@k, MRR, nDCG@k, and an NLI-based
                                      hallucination proxy
  ----------------------------------------------------------------------------

## Repository structure

``` text
legal_rag/
├── app.py                 # Streamlit interface
├── config.py              # Models, paths, and retrieval/verification settings
├── evaluate.py            # Retrieval ablation and grounding evaluation
├── generator.py           # LLM wrapper and citation-oriented prompt
├── grounding.py           # Sentence-level citation and entailment checks
├── ingest.py              # Document loading, chunking, and FAISS index creation
├── pipeline.py            # End-to-end retrieve → generate → verify workflow
├── retriever.py           # BM25, FAISS, RRF, and cross-encoder re-ranking
├── requirements.txt       # Python dependencies
├── data/
│   └── docs/              # Place source PDF/TXT files here
├── eval/
│   └── qa.jsonl           # Small example evaluation dataset
└── index/                 # Generated FAISS index and serialized chunks
```

The `index/` directory is created by ingestion and is not included as a
prebuilt index in this source package.

## Getting started

### 1. Requirements

-   Python 3.10 or newer is recommended.
-   `pip`
-   An OpenAI API key if using the default OpenAI provider, **or**
    sufficient local compute and memory for the Hugging Face model.
-   Internet access on first run to download the configured embedding,
    re-ranking, NLI, and optional local LLM models.

### 2. Create an environment and install dependencies

From the `legal_rag/` directory:

``` bash
python -m venv .venv
```

Activate it:

``` bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

``` bash
pip install -r requirements.txt
```

### 3. Configure the LLM

The default configuration uses OpenAI with `gpt-4o-mini`. Set the API
key before running the application.

``` bash
# macOS / Linux
export OPENAI_API_KEY="your_api_key_here"

# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"
```

Do not commit API keys to source control.

To use the Hugging Face provider instead, set `LLM_PROVIDER` to `hf`:

``` bash
# macOS / Linux
export LLM_PROVIDER=hf

# Windows PowerShell
$env:LLM_PROVIDER="hf"
```

The configured local model defaults to `Qwen/Qwen2.5-1.5B-Instruct`.
Model loading and hardware requirements depend on your machine. The
embedding, re-ranking, and NLI models are configured separately in
`config.py`.

## Add documents and build the index

1.  Put the legal documents you want to query into `data/docs/`. The
    loader supports `.pdf` and `.txt` files.
2.  Build the vector index and save the chunks:

``` bash
python ingest.py
```

The ingestion script reports the number of loaded pages/documents and
chunks, then writes the FAISS index and `chunks.pkl` under `index/`.

**Important:** Run ingestion again whenever you add, remove, or change
source documents. The index and stored chunks must correspond to the
same document collection.

## Run the application

### Streamlit UI

``` bash
streamlit run app.py
```

Enter a question in the interface. The app displays the answer and
source citations. Expand a source entry to inspect the cited passage,
source filename, page, and re-ranking score. If the grounding verifier
rejects generated sentences, the UI can show those removed sentences.

### Command-line query

``` bash
python pipeline.py "What is the limitation period for recovery of money?"
```

The command prints the final answer and the sources used.

## Evaluation

The repository includes a small illustrative dataset at `eval/qa.jsonl`.
It contains one answerable question tied to `example_act.txt` and one
unanswerable question. This is a smoke-test dataset, **not** a
statistically meaningful benchmark.

Run:

``` bash
python evaluate.py --qa eval/qa.jsonl
```

Optionally choose the evaluation cutoff:

``` bash
python evaluate.py --qa eval/qa.jsonl --k 5
```

The script evaluates retrieval modes:

-   `dense`
-   `bm25`
-   `hybrid`
-   `hybrid_rerank`

It reports:

-   **Recall@k:** whether a relevant source appears in the top-k
    retrieved results.
-   **MRR (Mean Reciprocal Rank):** rewards relevant sources appearing
    nearer the top.
-   **nDCG@k:** evaluates the ranking quality of relevant results.
-   **Raw hallucination rate:** an NLI-based proxy for unsupported
    sentences or answering questions marked unanswerable.
-   **Post-filter unanswerable-question rate:** how often the system
    returns an answer rather than abstaining on questions labelled
    unanswerable.

The grounding evaluation is a **proxy**, not proof that all answers are
correct. Manually annotate a larger and representative set of questions
before reporting performance claims. The evaluation script itself
explicitly recommends checking the NLI-based measure against a manually
annotated subset.

### Evaluation data format

Use one JSON object per line in the JSONL file:

``` json
{"question": "What is the limitation period?", "relevant_sources": ["example_act.txt"], "answerable": true}
{"question": "A question not answered by the documents?", "relevant_sources": [], "answerable": false}
```

`relevant_sources` contains filenames expected to be retrieved. For
unanswerable questions, use an empty list and set `answerable` to
`false`. Build a larger evaluation set with questions and labels that
match your own document collection.

## Configuration

The main settings are in `config.py`.

  -----------------------------------------------------------------------
  Setting                             Purpose
  ----------------------------------- -----------------------------------
  `chunk_size`                        Maximum chunk size used by the text
                                      splitter

  `chunk_overlap`                     Overlap between adjacent chunks

  `embedding_model`                   Model used to create dense
                                      embeddings

  `reranker_model`                    Cross-encoder used to re-rank
                                      candidates

  `nli_model`                         Model used for sentence-level
                                      entailment checks

  `top_k_dense`                       Candidate count from dense
                                      retrieval

  `top_k_bm25`                        Candidate count from BM25 retrieval

  `rrf_k`                             RRF rank constant

  `rerank_candidates`                 Number of fused candidates sent to
                                      the re-ranker

  `final_k`                           Number of passages passed to the
                                      LLM

  `min_rerank_score`                  Retrieval-score threshold below
                                      which the system abstains

  `entail_threshold`                  Minimum NLI entailment probability
                                      for a sentence to be accepted

  `llm_provider`                      `openai` or `hf`

  `openai_model` / `hf_model`         Configured generation model

  `max_new_tokens`                    Generation length limit
  -----------------------------------------------------------------------

Thresholds are starting configuration values, not universally calibrated
values. Tune them against a development set from your target documents
and inspect both missed answers and unsupported answers.

## Important implementation notes

-   Citation numbers refer to the ordered passages supplied to the LLM,
    not global document IDs.
-   The verifier checks that a sentence has a valid citation and that at
    least one cited passage meets the configured entailment threshold.
-   Unsupported sentences are removed; if no supported sentences remain,
    the pipeline abstains.
-   Source page metadata depends on the document loader and may use
    zero-based page numbering.
-   FAISS loading uses `allow_dangerous_deserialization=True` because
    the project loads its locally saved index. Only load index artifacts
    you created or otherwise trust.
-   The sample legal text in `data/docs/example_act.txt` is
    illustrative. Do not treat it as a statement of current law.

## Limitations and future work

Current limitations include:

-   The bundled evaluation dataset is very small and cannot establish
    general retrieval or answer quality.
-   NLI entailment is an imperfect proxy for legal correctness and can
    misclassify nuanced, conditional, or cross-document claims.
-   PDF extraction quality can affect retrieval, especially for scanned
    documents or complex layouts.
-   The default score and entailment thresholds require calibration for
    the target corpus.
-   The project does not replace legal review, legal advice, or
    verification against primary sources.

Potential extensions:

-   Add OCR for scanned PDFs and better handling of tables, footnotes,
    and section numbering.
-   Expand the evaluation set and add manually reviewed answer
    faithfulness labels.
-   Add document-level metadata filters, jurisdiction/date filters, and
    more robust citation rendering.
-   Track retrieval latency, generation latency, token usage, abstention
    rate, and reviewer-rated answer quality.
-   Add automated tests for citation parsing, abstention behavior, chunk
    metadata, and ranking logic.

## License

No license file is included in the provided project. Add a license only
after deciding how the code may be reused and distributed.
