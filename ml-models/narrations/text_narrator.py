import pdfplumber
from gtts import gTTS
import os
import numpy as np
import tensorflow as tf

class TextNarrator:
    def __init__(self, language='en'):
        # Set the language for narration
        self.language = language
        self.text = ""

    def load_text(self, text_or_file):
        # Load text from a file (PDF or text file) or direct string
        if os.path.isfile(text_or_file):
            if text_or_file.endswith('.pdf'):
                self.text = self._extract_text_from_pdf(text_or_file)
            else:
                with open(text_or_file, 'r', encoding='utf-8') as file:
                    self.text = file.read()
        else:
            self.text = text_or_file

    def _extract_text_from_pdf(self, pdf_path):
    # Helper method to extract text from PDF
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Check if text was extracted
                    text += page_text + "\n"
                else:
                    print("No text found on page.")
        return text


    def generate_audio(self):
        # Generate audio from the loaded text
        if self.text:
            self.tts = gTTS(text=self.text, lang=self.language, slow=False)
        else:
            raise ValueError("No text loaded. Please load text using load_text method.")
    
    def save_audio(self, file_path="output.mp3"):
        # Save the audio to a specified file path
        if hasattr(self, 'tts'):
            self.tts.save(file_path)
            print(f"Audio saved to {file_path}")
        else:
            raise ValueError("No audio generated. Please generate audio using generate_audio method.")

# Example Usage
if __name__ == "__main__":
    narrator = TextNarrator(language='en')
    pdf_path = "ml_models/pdfs/sample_chapter.pdf"  # Update this path to your PDF file
    narrator.load_text(pdf_path)

    narrator.generate_audio()
    narrator.save_audio("chapter_audio.mp3")
