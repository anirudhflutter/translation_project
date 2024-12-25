import base64
import io
import re
import sys
import tempfile
import threading
from django.shortcuts import render
from django.conf import settings
import numpy as np
import requests
import openai
import whisper
import queue
from testing_speech import synthesize_speech
from translator.models import TranslationSession
from googletrans import Translator
import google.generativeai as genai
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from .models import ConversationCategoryModel, TranslationSession
from django.views.decorators.http import require_POST
import datetime
from threading import Thread
import pyaudio
import wave
from loguru import logger
import threading
from django.http import JsonResponse
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize
import webrtcvad
import wave
import os
import noisereduce as nr
import struct
from google.cloud import texttospeech
from django.http import HttpResponse
from google.oauth2 import service_account
import json
from resemblyzer import VoiceEncoder, preprocess_wav

def generate_speech(request):
    # Set the German text
    text = request.POST.get('text', '').strip()
    # Load the service account credentials directly from the JSON key file
    credentials = service_account.Credentials.from_service_account_file('text_to_speech_google_cloud.json')

    # Initialize the client with the credentials
    client = texttospeech.TextToSpeechClient(credentials=credentials)

    # Set up the text input
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Set up the voice parameters (German voice)
    voice = texttospeech.VoiceSelectionParams(
        language_code='de-DE',  # German language
        name='de-DE-Wavenet-D',  # High-quality voice
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Set up audio file output format (MP3)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    # Synthesize the speech and get the response
    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    # Encode the audio content to base64
    audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')

    # Return the audio content as a JSON response
    return JsonResponse({"audio": audio_base64})
    
# Global variables
is_listening = False
recognized_text = ""
global_recognized_text = ""
is_processing = False  # New flag
lock = threading.Lock()
audio_queue = queue.Queue()

# for verifying enrolled user voice or not
DATA_DIR = 'enrolled_user/'
ENCODINGS_FILE = 'user_embeddings.npy'
REFERENCE_EMBEDDING_FILE = 'reference_embedding.npy'
LIVE_VERIFICATION_FILE = 'live_verification.wav'
SAMPLE_RATE = 16000
ENROLL_DURATION = 5  # seconds per enrollment sample
VERIFY_DURATION = 3  # seconds per verification
THRESHOLD = 0.75     # Similarity threshold for verification
encoder = VoiceEncoder()

def preprocess_audio(audio_data, sample_rate=16000):
    """
    Preprocess audio for better recognition accuracy:
    - Normalize volume
    - Reduce noise
    - Resample to 16kHz
    - Convert to mono
    - Save to a temporary WAV file
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_original:
        temp_original.write(audio_data.get_wav_data())
        temp_original_path = temp_original.name
    vad = webrtcvad.Vad()
    vad.set_mode(3)
    # Load audio with pydub
    sound = AudioSegment.from_file(temp_original_path)
    sound = normalize(sound)
    sound = sound.set_frame_rate(sample_rate).set_channels(1)
    sound = sound + 10  # Increase volume by 10 dB
    # Export to raw data for noise reduction
    raw_data = sound.raw_data
    samples = np.frombuffer(raw_data, dtype=np.int16)

    # Perform noise reduction
    reduced_noise = nr.reduce_noise(y=samples, sr=sample_rate)

    # Create a new AudioSegment from the reduced noise data
    reduced_sound = AudioSegment(
        reduced_noise.tobytes(),
        frame_rate=sample_rate,
        sample_width=reduced_noise.dtype.itemsize,
        channels=1
    )

    # Save the cleaned audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_cleaned:
        reduced_sound.export(temp_cleaned, format="wav")
        cleaned_audio_path = temp_cleaned.name

    # Remove the original temporary file
    os.remove(temp_original_path)

    return cleaned_audio_path


def is_user_enrolled(audio_data):
    try:
        # Convert the audio byte data to a file-like object
        audio_file = io.BytesIO(audio_data)  # Convert byte data to a file-like object
        
        # Save the received audio data temporarily as a WAV file
        with open(LIVE_VERIFICATION_FILE, 'wb') as temp_file:
            temp_file.write(audio_file.read())  # Write audio data to the temporary file
        
        # Process the saved WAV file (this is your preprocessing function)
        wav = preprocess_wav(LIVE_VERIFICATION_FILE)
        live_embedding = encoder.embed_utterance(wav)

        # Check if reference embedding exists
        if not os.path.exists(REFERENCE_EMBEDDING_FILE):
            print("Reference embedding not found. Please enroll first.")
            return JsonResponse({"error": "Reference embedding not found. Please enroll first."}, status=400)

        # Load the reference embedding
        reference_embedding = np.load(REFERENCE_EMBEDDING_FILE)

        # Compute similarity between live recording and reference embedding
        similarity = np.dot(reference_embedding, live_embedding) / (
            np.linalg.norm(reference_embedding) * np.linalg.norm(live_embedding)
        )
        
        print(f"Similarity Score: {similarity:.2f}")
        similarity = float(similarity)

        # Determine if the voice matches based on the threshold
        if similarity > THRESHOLD:
            return True
        else:
            return False
    
    except Exception as e:
        print(f"Error during verification: {e}")
        return JsonResponse({"error": f"Error during verification: {e}"}, status=500)

def listener(audio_queue):
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        print("Adjusting for background noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.dynamic_energy_threshold = True
        recognizer.energy_threshold = 900  # Default is 300, adjust as needed
        print("Listening continuously... (Press Stop to end)")
        while is_listening:
            try:
                print("Listening...")
                # Listen with short timeouts to periodically check is_listening
                audio_data = recognizer.listen(source, timeout=30, phrase_time_limit=20)
                #verify_user(audio_data, THRESHOLD)
                audio_queue.put(audio_data)
            except sr.WaitTimeoutError:
                print("No speech detected. Continuing...")
                continue
            except Exception as e:
                print(f"Error capturing audio: {e}")
                continue

response_flag = False
response_data = None
response_event = threading.Event()
event_queue = queue.Queue()

def stream_listener(audio_stream):
    """
    Listener function to process audio streams in real-time.
    """
    recognizer = sr.Recognizer()
    try:
        while is_listening:
            # Read from the audio stream
            audio_data = sr.AudioData(audio_stream.read(), sample_rate=16000, sample_width=2)

            # Recognize speech
            text = recognizer.recognize_google(audio_data, language="de-DE")
            print(f"Recognized text: {text}")
    except Exception as e:
        print(f"Error in stream_listener: {e}")

def processor(audio_queue):
    global recognized_text
    recognizer = sr.Recognizer()

    while is_listening or not audio_queue.empty():
        try:
            audio_data = audio_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            is_processing = True

            # Preprocess audio
            cleaned_audio_path = preprocess_audio(audio_data)

            # Load the cleaned audio file
            with sr.AudioFile(cleaned_audio_path) as source:
                processed_audio = recognizer.record(source)
            processed_audio_path = "processed_audio.wav"
            # Recognize speech using Google with German language
            text = recognizer.recognize_google(processed_audio, language="de-DE")
            print("You said:", text)
            if str(text).lower().__contains__("stoppen") or str(text).__contains__("Stopp") or str(text).lower().__contains__("stopp") or str(text).lower().__contains__("stop"):
                print("enteredinside")
                event_queue.put("stoppen_detected")
            if str(text).__contains__("finally"):
                print("finally")
                event_queue.put("finally_detected")
            if str(text).__contains__("beginnen") or str(text).__contains__("beginn") or str(text).__contains__("begin"):
                print("beginnen_detected")
                event_queue.put("beginnen_detected")
                clear_queue(audio_queue)
            while not audio_queue.empty():
                audio_queue.get()
            with lock:
                if not str(recognized_text).__contains__(text):
                    recognized_text += f"{text} "
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Error with the recognition service: {e}")
        except Exception as e:
            print(f"Error during transcription: {e}")
        finally:
            # Clean up the temporary file
            if os.path.exists(cleaned_audio_path):
                os.remove(cleaned_audio_path)
            is_processing = False
            audio_queue.task_done()

def continuous_speech_to_text():
    listener_thread = threading.Thread(target=listener, args=(audio_queue,))
    processor_thread = threading.Thread(target=processor, args=(audio_queue,))

    listener_thread.start()
    processor_thread.start()
    listener_thread.join()
    processor_thread.join()

def clear_queue(q):
    """
    Empties the given queue by removing all items.

    Args:
        q (queue.Queue): The queue to clear.
    """
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass  # Queue is now empty

def start_listening(request):
    global is_listening, recognized_text

    is_listening = True
    recognized_text = ""

    type = request.POST.get('type', '').strip()
    print(f"type {type}")
    if type == "continue_again":
        response_text = ""
    if type == "beginning":
        return JsonResponse({"status": "started_listening"})
    # Poll the event_queue in the main thread to check for 'gut'
    threading.Thread(target=continuous_speech_to_text, daemon=True).start()
    while True:
        print("1")
        try:
            event = event_queue.get(timeout=100)  # Wait for a maximum of 10 seconds
            print("2")
            if event == "finally_detected":
                is_listening = False
                return JsonResponse({"status": "Finally Stopped listening"})
            if event == "beginnen_detected":
                recognized_text = ""
            if event == "stoppen_detected":
                if not is_listening:
                    return JsonResponse({"error": "Not currently listening"}, status=400)
                # Wait until the processing of the current chunk is complete
                while is_processing:
                    datetime.time.sleep(0.1)  # Small delay to prevent high CPU usage
                # Wait until all audio in the queue is processed
                audio_queue.join()
                with lock:
                    response_text = recognized_text.strip()
                    recognized_text = ""
                    global_recognized_text = response_text
                response_text = str(response_text).lower().replace("stoppen","")
                response_text = str(response_text).lower().replace("stopp","")

                return JsonResponse({"status": "Stopped listening", "transcription": response_text})
        except queue.Empty:
            # Timeout reached without detecting "gut", respond accordingly
            return JsonResponse({"status": "No 'gut' detected within timeout"})



def stop_listening(request):
    global is_listening, recognized_text, is_processing
    if not is_listening:
        return JsonResponse({"error": "Not currently listening"}, status=400)
    
    # Wait until the processing of the current chunk is complete
    while is_processing:
        datetime.time.sleep(0.1)  # Small delay to prevent high CPU usage

    is_listening = False
    # Wait until all audio in the queue is processed
    audio_queue.join()

    with lock:
        response_text = recognized_text.strip()
        recognized_text = ""

    return JsonResponse({"status": "Stopped listening", "transcription": response_text})

@require_POST
def delete_session(request, session_id):
    session = get_object_or_404(TranslationSession, id=session_id)
    session.delete()
    return JsonResponse({'success': True, 'message': 'Translation session deleted successfully.'})
# Configure logging

# Hugging Face Inference API endpoint and headers
HF_API_URL = "https://api-inference.huggingface.co/models/gpt2"  # Choose an appropriate model
HF_API_TOKEN = settings.HUGGINGFACE_API_TOKEN  # Ensure this is set in your settings.py

hf_headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

openai.api_key = settings.OPENAI_API_KEY

def generate_suggestion(prompt):
    try:
        response = requests.post(
            HF_API_URL,
            headers=hf_headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_length": 100,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "do_sample": True,
                }
            }
        )
        response.raise_for_status()
        generated = response.json()
        print(f"generated {generated}")
        return generated[0]['generated_text'].strip()
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        logger.error(f"Response Content: {response.text}")
        return "I'm sorry, I couldn't generate a response at this time."
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return "I'm sorry, I couldn't generate a response at this time."

def index(request):
    return render(request, 'index.html')

def translate_view(request):
    translation = None
    suggestion = None
    user_input = ''
    language_direction = ''
    categories = ConversationCategoryModel.objects.all()
    if request.method == 'POST':
        is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest'
        language_direction = request.POST.get('language_direction', '').strip()
        category = request.POST.get('category', '').strip()
        user_input = request.POST.get('user_input', '').strip()
        print(f"user_input {is_ajax}")
        print(f"category {category}")
        if language_direction and user_input and is_ajax is True:
            translator = Translator()
            try:
                if language_direction == "de_to_en":
                    translation = translator.translate(user_input, src='de', dest='en').text
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    suggestion = model.generate_content(
                        f"I am learning german language so Provide a response in German and english which I should respond and ask to person who said this in german: \n\n{user_input}. "
                        "Important is format should be such that first line German:, then next line English:. keep it short and instead of example consider something yourself and give me to say. After replying to them tell me what to ask them to not make them bored with same conversation. keep it simple no extra suggestions also give tag like German: and English:. Make sure first to give full for German: then for English:, not like first german then english then again german english please not like that"
                    ).text
                elif language_direction == "en_to_de":
                    translation = translator.translate(user_input, src='en', dest='de').text
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    suggestion = model.generate_content(
                        f"I am learning English language so Provide a response in German and english which I should respond and ask to person who said this in English: \n\n{user_input}. "
                        "Important is format should be such that first line German:, then next line English:. keep it short also give tag like German: and English:"
                    ).text
                else:
                    translation = "Invalid language direction selected."
                    suggestion = ""

                print(f"suggestion {suggestion}")

                category_data = ConversationCategoryModel.objects.get(id=category)
                # Save the translation session to the database
                if translation and suggestion:
                    TranslationSession.objects.create(
                        language_direction=language_direction,
                        original_text=user_input,
                        category=category_data,
                        translated_text=translation,
                        suggested_response=suggestion
                    )
            except Exception as e:
                logger.error(f"Error during translation or suggestion generation: {e}")
                translation = "An error occurred during translation."
                suggestion = "An error occurred while generating a response."
            if is_ajax:
                if suggestion:
                    suggestion = str(suggestion).replace("*", "")
                if translation and suggestion:
                    match = re.search(r'German: (.*?)\s*English:', suggestion)
                    if match:
                        german_text = match.group(1)
                        print("German text:", german_text)
                    else:
                        print("No match found for German text.")
                        german_text =""
                    print("reached_here")
                    return JsonResponse({
                        "translation": translation,
                        "suggestion": suggestion,
                        "speak" : german_text
                    })
                else:
                    return JsonResponse({"error": "Translation or suggestion failed."}, status=400)

    # Handle non-AJAX GET request
    translation_sessions = TranslationSession.objects.all().order_by('-timestamp')[:50]
    for session in translation_sessions:
        if "English" in session.suggested_response:
            parts = session.suggested_response.split("English", 1)
            green_text = parts[0].strip()
            red_text = "English" + parts[1].strip()
            session.suggested_response = (
                f"<span style='color: green; font-weight: bold;'>{green_text}</span>"
                f"<br><br>"
                f"<span style='color: red; font-weight: bold;'>{red_text}</span>"
            )
        else:
            session.suggested_response = (
                f"<span style='color: green; font-weight: bold;'>{session.suggested_response.strip()}</span>"
            )

    return render(request, 'translator.html', {
        'translation': translation,
        'suggestion': suggestion,
        'user_input': user_input,
        "categories" : categories,
        'language_direction': language_direction,
        'translation_sessions': translation_sessions,
    })

def fetch_dashboard_data(request):
    if request.method == "GET":
        category_id = request.GET.get('category_id') 
        print(f"category_id {category_id}")
        if category_id:
            sessions = TranslationSession.objects.filter(category_id=category_id).select_related('category').order_by('-timestamp')
        else:
            sessions = TranslationSession.objects.all().select_related('category').order_by('-timestamp')
        data = [
            {
                "id": session.id,
                "language_direction": session.get_language_direction_display(),
                "original_text": session.original_text,
                "translated_text": session.translated_text,
                "category_name" : session.category.name,
                "suggested_response": session.suggested_response,
                "timestamp": session.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            }
            for session in sessions
        ]
        return JsonResponse({"sessions": data}, status=200)
    return JsonResponse({"error": "Invalid request method."}, status=400)