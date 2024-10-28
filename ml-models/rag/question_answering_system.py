import tensorflow as tf
import tensorflow_hub as hub
from transformers import AutoModelForCausalLM, AutoTokenizer
import pdfplumber
import numpy as np

class QuestionAnsweringSystem:
    def __init__(self, retriever_model_url='https://tfhub.dev/google/universal-sentence-encoder/4', generator_model='distilgpt2'):
        # Load the retrieval and generation models
        self.retriever = hub.load(retriever_model_url)
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.generator = AutoModelForCausalLM.from_pretrained(generator_model)

    def load_chapter(self, pdf_path):
        # Extract text from the PDF and create embeddings for retrieval
        chapter_text = self._extract_text_from_pdf(pdf_path)
        self.chunks, self.chunk_embeddings = self._embed_text(chapter_text)
    
    def _extract_text_from_pdf(self, pdf_path):
        # Helper to extract text from PDF
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text

    def _embed_text(self, text, chunk_size=200):
        # Split text into chunks and create embeddings for each
        sentences = text.split('.')
        chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
        embeddings = self.retriever(chunks).numpy()  # Use TensorFlow Hub model
        return chunks, embeddings
    
    def generate_answer(self, question):
        # Retrieve the most relevant chunk and generate an answer
        question_embedding = self.retriever([question]).numpy()
        scores = np.dot(question_embedding, self.chunk_embeddings.T)
        best_chunk_idx = np.argmax(scores)
        context = self.chunks[best_chunk_idx]
        
        # Pass the context and question to the generator model
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.generator.generate(**inputs, max_length=100, do_sample=True, top_k=50, top_p=0.95)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        return answer

    def evaluate_answer(self, answer, expected_answer):
        # For now, let's return a basic similarity score as a placeholder for evaluation
        answer_embedding = self.retriever([answer]).numpy()
        expected_answer_embedding = self.retriever([expected_answer]).numpy()
        return np.dot(answer_embedding, expected_answer_embedding.T).item()

# Example Usage
if __name__ == "__main__":
    qa_system = QuestionAnsweringSystem()
    pdf_path = "ml_models/pdfs/sample_chapter.pdf"  # Update this path to your PDF file
    qa_system.load_chapter(pdf_path)

    # Example question
    question = "What are the main concepts discussed in this chapter?"
    answer = qa_system.generate_answer(question)
    print(f"Answer: {answer}")

    # Example evaluation
    expected_answer = "Main concepts discussed include..."
    similarity_score = qa_system.evaluate_answer(answer, expected_answer)
    print(f"Similarity Score: {similarity_score}")
