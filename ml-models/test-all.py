# test_all.py

import os
from narrations.text_narrator import TextNarrator  # Adjust the path as necessary
from rag.question_answering_system import QuestionAnsweringSystem  # Adjust the path as necessary
from sample_tests.test_generator import TestGenerator  # Adjust the path as necessary
from chatbot.chapter_chatbot import ChapterChatbot  # Adjust the path as necessary

def test_text_narrator():
    print("Testing Text Narrator...")
    narrator = TextNarrator(language='en')
    pdf_path = "ml-models/pdfs/Science (1).pdf"
    narrator.load_text(pdf_path)
    narrator.generate_audio()
    narrator.save_audio("chapter_audio.mp3")
    print("Text Narrator test completed.\n")

def test_question_answering_system():
    print("Testing Question Answering System...")
    qa_system = QuestionAnsweringSystem()
    pdf_path = "ml-models/pdfs/Science (1).pdf"
    qa_system.load_chapter(pdf_path)
    question = "What is the main topic of the chapter?"
    answer = qa_system.generate_answer(question)
    print(f"Question: {question}\nAnswer: {answer}\n")
    print("Question Answering System test completed.\n")

def test_sample_test_generator():
    print("Testing Sample Test Generator...")
    test_generator = TestGenerator()
    pdf_path = "ml-models/pdfs/Science (1).pdf"
    test_generator.load_chapter(pdf_path)  # Assuming a method exists to load the chapter
    questions = test_generator.generate_questions(test_generator.chunks[0])  # Adjust as needed
    print("Generated Questions:")
    for q in questions:
        print(f"Q: {q['question']}\nOptions: {q['options']}\nCorrect: {q['correct_option']}\n")
    print("Sample Test Generator test completed.\n")

def test_chapter_chatbot():
    print("Testing Chapter Chatbot...")
    chatbot = ChapterChatbot()
    pdf_path = "ml-models/pdfs/Science (1).pdf"
    chatbot.load_chapter(pdf_path)
    user_query = "Can you summarize the chapter?"
    response = chatbot.get_response(user_query)
    print(f"User Query: {user_query}\nChatbot Response: {response}\n")
    print("Chapter Chatbot test completed.\n")

if __name__ == "__main__":
    test_text_narrator()
    test_question_answering_system()
    test_sample_test_generator()
    test_chapter_chatbot()
