# 🧠 LLM-Powered Resume Analyzer using RAG

An AI-powered Resume Analysis system built using Retrieval-Augmented Generation (RAG), FAISS vector search, and Large Language Models.

This application allows users to upload a PDF resume and perform intelligent analysis such as strengths identification, skill gap detection, resume scoring, and summarization.

---

## 🚀 Features

- 📄 Upload Resume (PDF)
- 🔎 Semantic Resume Search using FAISS
- 🧠 Retrieval-Augmented Generation (RAG)
- 📊 Resume Strength & Weakness Analysis
- 🎯 ML / AI Role Suitability Rating
- 📝 5-Line Resume Summary
- ⚡ Optimized with model caching
- 🌐 Deployable on HuggingFace Spaces

---

## 🏗️ Tech Stack

- Python
- Streamlit
- Sentence Transformers (all-MiniLM-L6-v2)
- FAISS (Vector Similarity Search)
- Transformers (LLM)
- PyPDF
- NumPy
- Torch

---

## 🧠 Architecture

1. Extract text from PDF
2. Split resume into chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS vector database
5. Retrieve top relevant chunks
6. Pass retrieved context to LLM
7. Generate intelligent response

This follows the Retrieval-Augmented Generation (RAG) pattern.

---

## 📦 Installation (Local Setup)

```bash
git clone https://github.com/your-username/LLM-Resume-Analyzer-RAG.git
cd LLM-Resume-Analyzer-RAG
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌍 Deployment

Deployed using HuggingFace Spaces (Streamlit SDK).

---

## 📌 Use Cases

- Resume evaluation for AI/ML roles
- Skill gap analysis
- ATS improvement insights
- Recruiter screening assistance

---

## 📈 Future Enhancements

- Job Description upload & similarity scoring
- ATS percentage scoring algorithm
- Role-based resume ranking
- LLM upgrade (Mistral / Llama / FLAN-T5)
- Cloud deployment with Docker

---

## 👩‍💻 Author

Prema H

---

If you found this useful, feel free to ⭐ the repository.
