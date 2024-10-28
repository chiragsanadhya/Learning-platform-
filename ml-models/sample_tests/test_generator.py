from transformers import AutoTokenizer, TFAutoModelForCausalLM
import random
import PyPDF2
import tensorflow as tf

class TestGenerator:
    def __init__(self, model_name="gpt2"):
        # Load the generative model for question and options creation using TensorFlow
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = TFAutoModelForCausalLM.from_pretrained(model_name)

    def extract_text_from_pdf(self, pdf_path):
        # Extract text from the provided PDF file
        text = ""
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def generate_questions(self, pdf_path, num_questions=5, question_type="MCQ"):
        # Extract text from the PDF
        chapter_text = self.extract_text_from_pdf(pdf_path)

        # Generate multiple questions and options from chapter text
        questions = []
        for _ in range(num_questions):
            question = self._generate_question(chapter_text)
            options, correct_option = self._generate_options(question, question_type)
            questions.append({
                "question": question,
                "options": options,
                "correct_option": correct_option
            })
        return questions
    
    def _generate_question(self, text):
        # Generate a question prompt based on the chapter text
        prompt = f"Generate a question based on the following text:\n{text}\nQuestion:"
        inputs = self.tokenizer(prompt, return_tensors="tf")  # Use TensorFlow tensors
        output = self.model.generate(**inputs, max_length=50, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
        question = self.tokenizer.decode(output[0], skip_special_tokens=True).split("Question:")[-1].strip()
        return question
    
    def _generate_options(self, question, question_type):
        # Generate answer options and determine correct ones
        prompt = f"Create 4 plausible answer choices for this question: {question}\nOptions:"
        inputs = self.tokenizer(prompt, return_tensors="tf")  # Use TensorFlow tensors
        output = self.model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95)
        options_text = self.tokenizer.decode(output[0], skip_special_tokens=True).split("Options:")[-1].strip()
        options = [opt.strip() for opt in options_text.split("\n") if opt.strip()]

        # Adjust the options and correct choices for MCQ or MSQ
        if question_type == "MCQ":
            correct_option = random.choice(options)
        else:  # MSQ
            correct_option = random.sample(options, k=2)

        return options, correct_option
    
    def format_test(self, questions):
        # Format questions and options in a structured format
        formatted_test = []
        for q in questions:
            formatted_test.append({
                "question": q["question"],
                "options": q["options"],
                "correct_option": q["correct_option"]
            })
        return formatted_test

# Example Usage
if __name__ == "__main__":
    test_generator = TestGenerator()
    pdf_path = "ml_models/pdfs/sample_chapter.pdf"  # Update this path to your PDF file
    questions = test_generator.generate_questions(pdf_path, num_questions=5, question_type="MCQ")
    formatted_test = test_generator.format_test(questions)

    for idx, q in enumerate(formatted_test):
        print(f"Q{idx + 1}: {q['question']}")
        for option in q['options']:
            print(f"- {option}")
        print(f"Correct Option: {q['correct_option']}\n")
