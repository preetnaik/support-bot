import logging
import random
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# ----------------- Logging Setup -----------------
logging.basicConfig(
    filename="support_bot_log.txt",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

# ----------------- SupportBotAgent -----------------
class SupportBotAgent:
    def __init__(self, document_path):
        # Load QA model from Hugging Face
        device = 0 if torch.cuda.is_available() else -1
        self.qa_model = pipeline(
            "question-answering",
            model="distilbert-base-uncased-distilled-squad",
            device=device
        )

        # Load embedder
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Load document (PDF)
        self.document_text = self.load_pdf(document_path)

        # Split into sections (paragraphs)
        self.sections = [s.strip() for s in self.document_text.split("\n\n") if s.strip()]

        if self.sections:
            # Create embeddings
            self.section_embeddings = self.embedder.encode(self.sections, convert_to_tensor=True)
        else:
            self.section_embeddings = None

        logging.info(f"Loaded document: {document_path} with {len(self.sections)} sections")

    # --------- Load PDF ---------
    def load_pdf(self, path):
        """Extract text from PDF file."""
        text = ""
        try:
            with open(path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            logging.info(f"Extracted text from {path}, length={len(text)} chars")
        except Exception as e:
            logging.error(f"Error loading PDF: {e}")
        return text or " "  # return at least a space to avoid empty document

    # --------- Retrieval ---------
    def find_relevant_section(self, query):
        """Find most relevant document section."""
        if not self.sections or self.section_embeddings is None:
            return "No relevant information available."

        try:
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, self.section_embeddings)[0]
            best_idx = int(similarities.argmax())
            logging.info(f"Best section index={best_idx} for query={query}")
            return self.sections[best_idx]
        except Exception as e:
            logging.error(f"Error finding relevant section: {e}")
            return "No relevant information available."

    # --------- Answering ---------
    def answer_query(self, query):
        """Answer query using QA model + context."""
        context = self.find_relevant_section(query)
        if not context or context.strip() == "":
            return "I don’t have enough information to answer that."

        try:
            result = self.qa_model(question=query, context=context)
            answer = result.get("answer", "").strip()
            if not answer:
                return "I don’t have enough information to answer that."
            return answer
        except Exception as e:
            logging.error(f"Error in QA model: {e}")
            return "Sorry, I couldn't process your question."

    # --------- Feedback Simulation ---------
    def get_feedback(self, response):
        feedback = random.choice(["not helpful", "too vague", "good"])
        logging.info(f"Feedback received: {feedback}")
        return feedback

    # --------- Adjust Responses ---------
    def adjust_response(self, query, response, feedback):
        if feedback == "too vague":
            context = self.find_relevant_section(query)
            return f"{response} (Extra info: {context[:150]}...)"  # add more context
        elif feedback == "not helpful":
            return self.answer_query(query + " Please explain in more detail.")
        return response

    # --------- Run Bot ---------
    def run(self, queries):
        for query in queries:
            logging.info(f"Processing query: {query}")
            response = self.answer_query(query)
            print(f"\nInitial Response to '{query}': {response}")

            # Feedback loop (max 2 iterations)
            for _ in range(2):
                feedback = self.get_feedback(response)
                if feedback == "good":
                    break
                response = self.adjust_response(query, response, feedback)
                print(f"Updated Response to '{query}': {response}")


# ----------------- Main -----------------
if __name__ == "__main__":
    bot = SupportBotAgent(r"C:\Users\preet\OneDrive\Documents\support bot\Serri Doc.pdf")
    
    print("Support Bot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
        answer = bot.answer_query(user_input)
        print("Bot:", answer)
