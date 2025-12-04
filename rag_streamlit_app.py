import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
import numpy as np
import faiss
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

st.title("Simple RAG PDF Q&A (OpenAI + FAISS)")
st.write("Upload a PDF and ask questions about its content.")

# Check for API key
if not API_KEY:
    st.error("No API key found in `.env` file. Please add OPENAI_API_KEY.")
else:
    client = OpenAI(api_key=API_KEY)

# ---- Helper functions ----
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks


def embed_texts(client, texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]


def retrieve(query, chunks, index, client, k=3):
    query_embedding = embed_texts(client, [query])[0]
    D, I = index.search(
        np.array([query_embedding]).astype("float32"),
        k
    )
    return [chunks[i] for i in I[0]]


def answer_question(client, context, question):
    prompt = f"""
Use ONLY the context below to answer the question.

Context:
{context}

Question: {question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# ---- PDF Upload UI ----
uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_pdf and API_KEY:
    st.success("PDF uploaded successfully!")

    with st.spinner("Extracting PDF text..."):
        text = load_pdf(uploaded_pdf)

    with st.spinner("Chunking text..."):
        chunks = chunk_text(text)

    with st.spinner("Creating embeddings..."):
        embeddings = embed_texts(client, chunks)

    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    st.success("PDF processed! Ask your questions.")

    question = st.text_input("Ask a question about the PDF:")

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    if st.button("Submit Question"):
        st.session_state.submitted = True

    if st.session_state.submitted:
        if question.strip() == "":
            st.warning("Please type a question first.")
        else:
            with st.spinner("Thinking..."):
                retrieved_chunks = retrieve(question, chunks, index, client)
                context = "\n\n".join(retrieved_chunks)
                answer = answer_question(client, context, question)

            st.subheader("Answer:")
            st.write(answer)

            st.subheader("Retrieved Context (for transparency):")
            st.write(context)

        # reset state for next question
        st.session_state.submitted = False

    else:
        st.info("Upload a PDF to begin.")