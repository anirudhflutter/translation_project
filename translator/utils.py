import speech_recognition as sr
from googletrans import Translator
from transformers import pipeline

def convert_speech_to_text(audio_file_path, language="en-US"):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data, language=language)
    return text

def translate_text(text, src_language="en", dest_language="es"):
    translator = Translator()
    translated = translator.translate(text, src=src_language, dest=dest_language)
    return translated.text


def generate_question(context, language="en"):
    question_generator = pipeline("text2text-generation", model="t5-small")
    question = question_generator(f"generate question: {context}", max_length=50, num_return_sequences=1)
    return question[0]["generated_text"]