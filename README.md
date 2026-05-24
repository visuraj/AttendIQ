# 🕐 AttendIQ — Smart Employee Attendance Q&A

A RAG-powered attendance analytics system. Upload attendance records + HR policy,
then ask natural language questions and get cited answers with:
- 📋 **Attendance Evidence** — employee records (scheduled vs actual, late/early flags)
- 📜 **Policy Evidence** — the HR policy section that applies

> Built for the Hack-AI-thon screening assignment.

---

## 🎬 Demo
[▶ Watch demo video]  https://drive.google.com/file/d/14CtyG-6SAEqb88lZN7E9acryW6KK89bp/view?usp=sharing

---

## 🏗️ Architecture

```
attendance_data.csv  +  attendance_policy.md
         │
         ▼
  Record Classifier (late / early / absent / on time)
         │
         ▼
  Natural Language Conversion per record
         │
         ▼
  text-embedding-3-small → ChromaDB
         │
    User Query
         │
         ▼
  Similarity Retrieval (top 10)
  → split into attendance docs + policy docs
         │
         ▼
  GPT-4o-mini (temperature=0) + context prompt
         │
         ▼
  Answer  +  Attendance Evidence  +  Policy Evidence
```

---

## ⚙️ Setup

```bash
git clone https://github.com/visuraj/AttendIQ
cd attendiq
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

Open http://localhost:8501 — enter API key(Your Google_API)→ click Load & Index Data → ask questions.

---

## 📁 Structure

```
attendiq/
├── src/
│   ├── app.py                 # Streamlit UI
│   ├── rag_pipeline.py        # Core RAG logic
│   ├── attendance_data.csv    # Sample data (30 records, 10 employees)
│   └── attendance_policy.md   # HR policy document
├── requirements.txt
└── README.md
```

---

## 💡 Example Questions

- *"Who was late on 2026-05-18?"*
- *"Which employees left early this week?"*
- *"Show Arjun Nair's attendance record"*
- *"Who was absent and what does policy say about it?"*
- *"Summarize Engineering department attendance"*

---

## 🧩 Engineering Challenge

**Mixing structured + unstructured data in one vector index.**
Attendance records are structured (CSV rows with times) but policy is unstructured prose.
Embedding them in the same ChromaDB collection and tagging each document with
`source: "attendance"` or `source: "policy"` lets the retriever fetch both types
simultaneously, which the LLM then uses to give policy-aware answers with data citations.

---

## 📜 License
MIT
