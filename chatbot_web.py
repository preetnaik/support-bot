import streamlit as st
import logging
import random
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# ----------------- SupportBotAgent -----------------
class SupportBotAgent:
    def __init__(self, document_path):
        device = 0 if torch.cuda.is_available() else -1
        self.qa_model = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad",
            device=device
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.document_text = self.load_pdf(document_path)
        self.sections = [s.strip() for s in self.document_text.split("\n\n") if s.strip()]
        self.section_embeddings = self.embedder.encode(self.sections, convert_to_tensor=True)

    def load_pdf(self, path):
        text = ""
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text

    def find_relevant_section(self, query):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, self.section_embeddings)[0]
        best_idx = int(similarities.argmax())
        return self.sections[best_idx]

    def answer_query(self, query):
        context = self.find_relevant_section(query)
        if not context:
            return "I don’t have enough information to answer that."
        result = self.qa_model(question=query, context=context)
        return result.get("answer", "").strip() or "I don’t have enough information to answer that."


# ----------------- Streamlit UI -----------------
st.title("📖 Customer Support Bot (PDF-trained)")
st.write("Upload a FAQ / Product Manual PDF and ask questions!")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF uploaded. Ready to answer queries.")
    bot = SupportBotAgent("temp.pdf")

    query = st.text_input("Ask a question:")
    if query:
        response = bot.answer_query(query)
        st.markdown(f"**Answer:** {response}")
