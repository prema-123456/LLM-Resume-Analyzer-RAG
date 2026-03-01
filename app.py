import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

st.title("🧠 LLM Resume Analyzer using RAG")

# Upload Resume
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

# Upload Job Description (Optional)
jd_file = st.file_uploader("Upload Job Description (Optional)", type="pdf")

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


if uploaded_file:

    # Extract Resume Text
    resume_text = extract_text_from_pdf(uploaded_file)

    # Extract JD Text (if uploaded)
    jd_text = ""
    if jd_file:
        jd_text = extract_text_from_pdf(jd_file)

    # Split text into chunks
    def split_text(text, chunk_size=500):
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks

    chunks = split_text(resume_text)

    # Create embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    # Store in FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    st.success("Resume indexed successfully ✅")

    # -------- Semantic Similarity (Resume vs JD) --------
    if jd_text:
        jd_embedding = model.encode([jd_text])
        resume_embedding = model.encode([resume_text])

        similarity = cosine_similarity(
            resume_embedding, jd_embedding
        )[0][0]

        st.subheader("📊 Resume-JD Match Score")
        st.write(f"Semantic Similarity: {round(similarity * 100, 2)}%")

    # -------- Analysis Section --------
    st.divider()
    st.header("📊 AI Resume Analysis")

    questions = [
        "What are the candidate's strengths?",
        "What skills are missing for AI Developer role?",
        "Rate this resume out of 10 for ML Engineer role.",
        "Summarize this resume in 5 lines."
    ]

    selected_question = st.selectbox("Select Analysis Type", questions)

analyze_button = st.button("🔍 Generate Analysis")

if analyze_button:

    query_embedding = model.encode([selected_question])
    D, I = index.search(np.array(query_embedding), k=3)

    retrieved_chunks = [chunks[i] for i in I[0]]
    context = " ".join(retrieved_chunks)

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    prompt = f"""
    You are an AI recruiter.

    Based on the resume below:

    {context}

    Answer the following clearly and professionally:

    {selected_question}
    """

    response = generator(prompt, max_length=300)

    st.subheader("🧠 AI Response")
    st.write(response[0]['generated_text'])

       