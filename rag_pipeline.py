import os, csv, google.generativeai as genai
from datetime import datetime, time as dtime
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

def _parse(t):
    try:
        return datetime.strptime(t.strip(), "%H:%M").time() if t.strip() else None
    except:
        return None

def _diff(t1, t2):
    return (t1.hour*60+t1.minute)-(t2.hour*60+t2.minute)

def classify(row):
    flags, late_min, early_min = [], 0, 0
    if row["status"] == "absent":
        return {**row, "flags": ["absent"], "late_by": 0, "early_by": 0, "verdict": "absent"}
    ai = _parse(row.get("actual_in", ""))
    ao = _parse(row.get("actual_out", ""))
    si = _parse(row.get("scheduled_in", ""))
    so = _parse(row.get("scheduled_out", ""))
    if ai and si:
        d = _diff(ai, si)
        if d > 10:
            late_min = d
            flags.append(f"late by {d} minutes")
        elif d < 0:
            flags.append(f"arrived {abs(d)} minutes early")
    if ao and so:
        d = _diff(ao, so)
        if d < -15:
            early_min = abs(d)
            flags.append(f"left {abs(d)} minutes early")
        elif d > 30:
            flags.append(f"overtime {d} minutes")
    verdict = "on time"
    if late_min > 10 and early_min > 15:
        verdict = "late arrival & early departure"
    elif late_min > 10:
        verdict = "late"
    elif early_min > 15:
        verdict = "early departure"
    return {**row, "flags": flags if flags else ["on time"],
            "late_by": late_min, "early_by": early_min, "verdict": verdict}

def record_to_text(r):
    flags_str = ", ".join(r["flags"])
    if r["status"] == "absent":
        return (f"{r['name']} ({r['department']}) on {r['date']}: "
                f"scheduled {r['scheduled_in']}-{r['scheduled_out']}, status absent. Flags: absent.")
    return (f"{r['name']} ({r['department']}) on {r['date']}: "
            f"scheduled {r['scheduled_in']}-{r['scheduled_out']}, "
            f"actual {r['actual_in']}-{r['actual_out']}, "
            f"status {r['status']}. Flags: {flags_str}.")

class AttendIQPipeline:
    PERSIST_DIR = ".chroma_attendiq"

    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["GOOGLE_API_KEY"] = api_key
        genai.configure(api_key=api_key)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self.records = []

    def load_csv(self, csv_path):
        self.records = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.records.append(classify(row))
        return len(self.records)

    def load_policy(self, policy_path):
        with open(policy_path, encoding="utf-8") as f:
            return f.read()

    def build_index(self, csv_path, policy_path):
        n = self.load_csv(csv_path)
        policy_text = self.load_policy(policy_path)
        docs = []
        for r in self.records:
            docs.append(Document(
                page_content=record_to_text(r),
                metadata={
                    "source": "attendance",
                    "doc_id": f"attendance:{r['employee_id']}:{r['date']}",
                    "employee_id": r["employee_id"],
                    "name": r["name"],
                    "department": r["department"],
                    "date": r["date"],
                    "verdict": r["verdict"],
                    "flags": ", ".join(r["flags"]),
                    "scheduled_in": r.get("scheduled_in", ""),
                    "scheduled_out": r.get("scheduled_out", ""),
                    "actual_in": r.get("actual_in", ""),
                    "actual_out": r.get("actual_out", ""),
                    "status": r["status"],
                }
            ))
        for i, section in enumerate(policy_text.split("\n## ")):
            if section.strip():
                docs.append(Document(
                    page_content=section.strip(),
                    metadata={
                        "source": "policy",
                        "doc_id": f"attendance-policy.md#{i}",
                        "section": section.split("\n")[0].strip("# "),
                    }
                ))
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=self.PERSIST_DIR,
        )
        return {"attendance_records": n, "total_docs": len(docs)}

    def query(self, question):
        if not self.vectorstore:
            return {"answer": "Please load data first.", "attendance": [], "policy": []}
        docs = self.vectorstore.as_retriever(search_kwargs={"k": 10}).invoke(question)
        att_docs = [d for d in docs if d.metadata.get("source") == "attendance"]
        pol_docs = [d for d in docs if d.metadata.get("source") == "policy"]
        att_context = "\n".join(d.page_content for d in att_docs[:5])
        pol_context = "\n".join(d.page_content for d in pol_docs[:3])
        prompt = (
            "You are an HR attendance assistant. Answer using only the data below. "
            "Be concise. Mention names, dates, and flags.\n\n"
            f"Attendance Records:\n{att_context}\n\n"
            f"HR Policy:\n{pol_context}\n\n"
            f"Question: {question}\nAnswer:"
        )
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            answer = model.generate_content(prompt).text
        except Exception as e:
            answer = f"LLM error: {e}"

        att_evidence, seen = [], set()
        for d in att_docs[:5]:
            did = d.metadata.get("doc_id", "")
            if did not in seen:
                seen.add(did)
                att_evidence.append({
                    "doc_id": did,
                    "name": d.metadata.get("name", ""),
                    "department": d.metadata.get("department", ""),
                    "date": d.metadata.get("date", ""),
                    "scheduled_in": d.metadata.get("scheduled_in", ""),
                    "scheduled_out": d.metadata.get("scheduled_out", ""),
                    "actual_in": d.metadata.get("actual_in", ""),
                    "actual_out": d.metadata.get("actual_out", ""),
                    "status": d.metadata.get("status", ""),
                    "flags": d.metadata.get("flags", ""),
                    "verdict": d.metadata.get("verdict", ""),
                    "snippet": d.page_content,
                })

        pol_evidence, seen_p = [], set()
        for d in pol_docs[:3]:
            did = d.metadata.get("doc_id", "")
            if did not in seen_p:
                seen_p.add(did)
                pol_evidence.append({
                    "doc_id": did,
                    "section": d.metadata.get("section", "Policy"),
                    "snippet": d.page_content[:300],
                })

        return {"answer": answer, "attendance": att_evidence, "policy": pol_evidence}

    def get_summary(self):
        if not self.records:
            return {}
        total = len(self.records)
        absent = sum(1 for r in self.records if r["status"] == "absent")
        late = sum(1 for r in self.records if r["late_by"] > 10)
        early = sum(1 for r in self.records if r["early_by"] > 15)
        return {"total": total, "on_time": total-absent-late-early,
                "late": late, "early_departure": early, "absent": absent}
