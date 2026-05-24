# ============================================================
# Section 3 — Debug Challenge: Fixed RAG Pipeline
# Original code had 8 bugs. All identified and fixed below.
# ============================================================

# BUG 6 FIX — Deprecated import paths (causes ImportError in LangChain v0.2+)
# ❌ from langchain.embeddings import OpenAIEmbeddings
# ❌ from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# ============================================================
# ORIGINAL BUGGY CODE (for reference — DO NOT USE)
# ============================================================

# BUG 1: chunk_overlap (500) > chunk_size (100) → ValueError crash
# splitter = CharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=500       ← CRASH: overlap must be < chunk_size
# )

# BUG 2: gpt-4 is a chat model, NOT an embedding model → API 404 error
# embeddings = OpenAIEmbeddings(model="gpt-4")

# BUG 3: temperature=1.0 → random hallucinated answers every time
# llm = ChatOpenAI(model="gpt-4", temperature=1.0)

# BUG 4: k=50 → context overflow + huge cost per query
# retriever=vectorstore.as_retriever(search_kwargs={"k": 50})

# BUG 5: return_source_documents=False → citations silently dropped
# return_source_documents=False

# BUG 7: No error handling → crashes on any API error or rate limit
# result = qa_chain.run("What is our refund policy?")

# BUG 8: .run() deprecated → use .invoke() instead
# qa_chain.run(...)

# ============================================================
# FIXED CODE
# ============================================================

# Load documents
loader = TextLoader("company_docs.txt")
docs = loader.load()

# BUG 1 FIX — chunk_overlap must be LESS than chunk_size
# ❌ chunk_size=100, chunk_overlap=500
# ✅ chunk_size=512, chunk_overlap=50
splitter = CharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# BUG 2 FIX — Use correct embedding model (not a chat model)
# ❌ OpenAIEmbeddings(model="gpt-4")
# ✅ OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings)

# BUG 3 FIX — temperature=0 for deterministic, factual answers
# ❌ ChatOpenAI(model="gpt-4", temperature=1.0)
# ✅ ChatOpenAI(model="gpt-4", temperature=0)
llm = ChatOpenAI(model="gpt-4", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,

    # BUG 4 FIX — k=4 is sufficient; k=50 overflows the context window
    # ❌ search_kwargs={"k": 50}
    # ✅ search_kwargs={"k": 4}
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 4}
    ),

    # BUG 5 FIX — Must be True to return citations
    # ❌ return_source_documents=False
    # ✅ return_source_documents=True
    return_source_documents=True
)

# BUG 7 + BUG 8 FIX — Add error handling and use .invoke() not .run()
# ❌ result = qa_chain.run("What is our refund policy?")
# ✅ Use try/except + .invoke()
try:
    result = qa_chain.invoke({"query": "What is our refund policy?"})
    print("Answer:", result["result"])

    # Now we can actually show citations (only works because BUG 5 was fixed)
    print("\nSources:")
    for doc in result["source_documents"]:
        print(f"  - {doc.metadata.get('source', 'unknown')}")

except Exception as e:
    print(f"Query failed: {e}")


# ============================================================
# SUMMARY OF ALL 8 BUGS
# ============================================================
#
# Bug 1: chunk_overlap (500) > chunk_size (100)
#        → Causes: ValueError crash, program never runs
#        → Fix: chunk_size=512, chunk_overlap=50
#
# Bug 2: OpenAIEmbeddings(model="gpt-4")
#        → Causes: API 404 error at runtime (gpt-4 is not an embedding model)
#        → Fix: model="text-embedding-3-small"
#
# Bug 3: ChatOpenAI(temperature=1.0)
#        → Causes: Hallucinated, random, inconsistent answers silently
#        → Fix: temperature=0
#
# Bug 4: search_kwargs={"k": 50}
#        → Causes: Context window overflow under load, huge API cost
#        → Fix: k=4
#
# Bug 5: return_source_documents=False
#        → Causes: All citation data silently dropped, answers unverifiable
#        → Fix: return_source_documents=True
#
# Bug 6: from langchain.embeddings / langchain.chat_models
#        → Causes: ImportError in LangChain v0.2+
#        → Fix: from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#
# Bug 7: No try/except around qa_chain call
#        → Causes: Any API error or rate limit crashes the entire service
#        → Fix: Wrap in try/except Exception
#
# Bug 8: qa_chain.run() deprecated
#        → Causes: Breaks in future LangChain, fails to return source docs
#        → Fix: qa_chain.invoke({"query": question})
#
# ============================================================
