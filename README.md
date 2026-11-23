# Claim Check AI 🔍🤖

Claim Check AI is an automated fact-checking and claim verification system. It leverages Semantic Search (Dense Retrieval) and Natural Language Inference (NLI) to analyze a written claim against a collection of scientific facts, ultimately rendering a final verdict: **SUPPORTED**, **REFUTED**, or **NOT ENOUGH EVIDENCE**.

## 🚀 Project Overview

* **Input:** A single written claim (e.g., *"Jupiter is the largest planet in the Solar System."*).
* **Evidence Source:** A curated database of science facts (`ClaimCheckAI_practice_1000.csv`).
* **Methodology:**
  1. **Deduplication:** Filters out redundant evidence passages to streamline the retrieval index.
  2. **Semantic Retrieval:** Uses a SentenceTransformer model (`all-MiniLM-L6-v2`) to encode and retrieve the top-$k$ most semantically relevant evidence facts.
  3. **Cross-Encoder Reasoning:** Passes the claim and retrieved evidence pairs through a pre-trained NLI model (`cross-encoder/nli-deberta-v3-small`) to predict entailment (Supports), contradiction (Refutes), or neutral relationships.
  4. **Threshold Filtering:** Ignores irrelevant evidence with low semantic similarity to avoid false positive matches.
* **Output:** A single, definitive consolidated verdict with the corresponding supporting or refuting evidence and its source URL.

---

## 🛠️ Tech Stack & Key Libraries

* **Data Processing:** `pandas`, `numpy`
* **Retrieval (Embedding Model):** `sentence-transformers` (specifically `all-MiniLM-L6-v2` for dense embeddings)
* **Re-ranking & Classification:** `transformers` & `torch` (specifically `cross-encoder/nli-deberta-v3-small` NLI pipeline)
* **Text Representation:** `scikit-learn` (TF-IDF vectorizer utilized for initial baseline comparisons)
* **Application Framework:** `streamlit` (Optional for building a web application dashboard)

---

## 📥 Installation & Setup

Make sure you have Python installed, then install the required dependencies:

```bash
pip install pandas numpy sentence-transformers transformers torch scikit-learn streamlit
```

---

## 📖 How It Works (Pipeline Steps)

### 1. Data Processing & Deduplication
We load the initial dataset containing 1,000 facts and deduplicate the evidence field to build an indexed knowledge base of unique science facts.

```python
df_unique = df.drop_duplicates(subset=["evidence"]).reset_index(drop=True)
```

### 2. Semantic Representation
We generate dense vector representations of the unique evidence passages using `SentenceTransformer`:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
evidence_embeddings = model.encode(df_unique["evidence"].tolist(), convert_to_tensor=True)
```

### 3. Verdict Assessment & Inference
For any input claim, we execute a semantic search, filter out results below a relevance threshold (e.g., `0.5`), and pass the top matching pairs to the DeBERTa-v3 NLI pipeline to establish the ground truth verdict:

```python
# Execution Example
final_verdict("Jupiter is the largest planet in the Solar System.")
```

**Output:**
```text
Final verdict: SUPPORTED
Evidence: Jupiter is the largest planet in the Solar System.
Source: [redacted link]
```

---

## 📊 Dataset Distribution

The practice dataset consists of 1,000 instances with the following categorical labels:

* **SUPPORTS:** 487
* **REFUTES:** 469
* **NOT ENOUGH INFO:** 44

---

## 🔮 Future Enhancements

* **Streamlit App Integration:** Convert the Python functions into a user-friendly UI dashboard.
* **Hybrid Search:** Implement a combination of BM25 (keyword matching) and Dense Retrieval (embeddings) for more robust evidence retrieval.
* **Extended Evaluation:** Produce classification reports and confusion matrices using the train/test split to systematically measure validation accuracy.
