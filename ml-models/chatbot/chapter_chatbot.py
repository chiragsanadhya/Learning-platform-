import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import faiss
import PyPDF2

class ChapterChatbot:
    def __init__(self, embed_model_url="https://tfhub.dev/google/universal-sentence-encoder/4"):
        # Load embedding model from TensorFlow Hub
        self.embed_model = hub.load(embed_model_url)
        
        # FAISS index for similarity search
        self.index = None
        self.chapter_embeddings = []
        self.context_texts = []

    def extract_text_from_pdf(self, pdf_path):
        # Extract text from the provided PDF file
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def load_chapter(self, pdf_path):
        # Load chapter from PDF and compute embeddings
        chapter_text = self.extract_text_from_pdf(pdf_path)
        sections = self._split_text(chapter_text)
        embeddings = [self._compute_embedding(section) for section in sections]
        
        self.chapter_embeddings = np.vstack(embeddings)
        self.context_texts = sections
        
        # Build FAISS index for quick retrieval
        dimension = embeddings[0].shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.chapter_embeddings)

    def _split_text(self, text, max_length=512):
        # Split text into smaller sections for embeddings
        sentences = text.split(". ")
        splits, current_split = [], ""
        
        for sentence in sentences:
            if len((current_split + sentence).split()) < max_length:
                current_split += sentence + ". "
            else:
                splits.append(current_split.strip())
                current_split = sentence + ". "
                
        if current_split:
            splits.append(current_split.strip())
        
        return splits

    def _compute_embedding(self, text):
        # Compute embedding for text section using Universal Sentence Encoder
        embedding = self.embed_model([text]).numpy()
        return embedding

    def get_response(self, user_query):
        # Retrieve closest chapter section to the query
        query_embedding = self._compute_embedding(user_query)
        _, indices = self.index.search(query_embedding, k=1)
        context = self.context_texts[indices[0][0]]
        
        # Generate a basic response with the context
        response = f"I found some information in the chapter related to your question: {context}"
        
        return response

    def log_interaction(self, user_query, response):
        # Placeholder for logging the interaction
        print(f"User Query: {user_query}\nBot Response: {response}\n")

# Example Usage
if __name__ == "__main__":
    chatbot = ChapterChatbot()
    pdf_path = "ml_models/pdfs/sample_chapter.pdf"  # Update this path to your PDF file
    chatbot.load_chapter(pdf_path)

    # Example user query
    user_query = "What are the main concepts discussed in this chapter?"
    response = chatbot.get_response(user_query)
    chatbot.log_interaction(user_query, response)
