import base64
import json
import re
from channels.generic.websocket import AsyncWebsocketConsumer
from django.http import JsonResponse
from googletrans import Translator
import speech_recognition as sr
import tempfile
import os
import django
from django.conf import settings
import subprocess
import google.generativeai as genai
from asgiref.sync import sync_to_async

def send_data_for_bot_conversation(transcription,language,category_id,level,is_start):
            print(f"transcription {transcription}")
            print(f"language {language}")
            print(f"category_id {category_id}")
            print(f"level {level}")
            print(f"is_start {is_start}")
            try:
                genai.configure(api_key="AIzaSyBiIULy6YCD_1eNFdf8MSOJC7GVwM_GBiQ")
                model = genai.GenerativeModel("gemini-1.5-flash")
                if is_start:
                    suggestion = model.generate_content(f"I am learning {language}. I want to start conversation with you of {level} level related to {category_id}. Please give me a sentence I should say to you. Do not give me 1 word give me a good sentence and then suggest me what I should reply back to you. See just say me sentence of conversation of yours and what I should say thats it. Format should be such that Yours Sentence:, next line My Sentence:, thats it").text
                else:
                    suggestion = model.generate_content(
                        f"""
                        I am learning {language}. Please create a new and unique conversation for me at the {level} level. 
                        The conversation should relate to {category_id}. Your response must strictly follow this format:

                        Yours Sentence: [Provide a unique and engaging sentence here]
                        My Sentence: [Provide a meaningful and contextually appropriate reply here]

                        Make sure the conversation is unique every time, engaging, and adheres to the level of difficulty specified.
                        """
                    ).text
                print(f"suggestion {suggestion}")
                return suggestion
            except Exception as e:
                print(f"exception in google model {e}")
                suggestion = "An error occurred while generating a response."


def translator_suggestion(transcription,preferred_language,buddy_language,category_id,is_translation):
            print(f"transcription {transcription}")
            print(f"preferred_language {preferred_language}")
            print(f"category_id {category_id}")
            translator = Translator()
            try:
                genai.configure(api_key="AIzaSyBiIULy6YCD_1eNFdf8MSOJC7GVwM_GBiQ")
                translation = translator.translate(transcription, src= buddy_language, dest=preferred_language).text
                print(f"translation {translation}")
                model = genai.GenerativeModel("gemini-1.5-flash")
                try:
                    suggestion = model.generate_content(
                        f"I am learning {buddy_language} language so Provide a response in {buddy_language} and {preferred_language} which I should respond and ask to person who said this in german: \n\n{transcription}. "
                        f"Important is format should be such that first line {buddy_language}:, then next line {preferred_language}:. keep it short and instead of example consider something yourself and give me to say. After replying to them tell me what to ask them to not make them bored with same conversation. keep it simple no extra suggestions also give tag like {buddy_language}: and {preferred_language}:. Make sure first to give full for {buddy_language}: then for {preferred_language}:, not like first {buddy_language} then {preferred_language}"
                    ).text
                    print(f"suggestion {suggestion}")
                except Exception as e:
                    print(f"gemini error {e}")
                if suggestion:
                    suggestion = str(suggestion).replace("*", "")
                if translation and suggestion:
                    pattern = fr'{re.escape(buddy_language)}: (.*?)\s*{re.escape(preferred_language)}:'
                    match = re.search(pattern, suggestion)
                    print(f"match123 {match}")
                    if match:
                        german_text = match.group(1)
                        print("German text:", german_text)
                    else:
                        print("No match found for German text.")
                    german_text =suggestion
                    print("reached_here")
                    return [translation,suggestion]
                else:
                    return []
            except Exception as e:
                translation = "An error occurred during translation."
                suggestion = "An error occurred while generating a response."


class AudioStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        from translator.models import TranslationSession
        print(f"text_data {text_data}")
        text_data_json = json.loads(text_data)
        is_translation = text_data_json.get("is_translation")
        transcription = text_data_json.get("transcription")
        category_id = text_data_json.get("category")
        action = text_data_json.get("action")
        """if is_translation:
            language = text_data_json.get("language")
            level = text_data_json.get("level")
            is_start = text_data_json.get("is_start")
            suggested_response = send_data_for_bot_conversation(transcription,language,category_id,level,is_start)
            print(f"suggested_response {suggested_response}")
                        # Save data into the database
            if suggested_response:
                await self.send(text_data=json.dumps({
                    'suggestion': suggested_response,  # Just an example
                }))"""
        #else:
        preferred_language = text_data_json.get("preferred_language")
        buddy_language = text_data_json.get("buddy_language")
        if action == 'translate':
            answer = translator_suggestion(transcription,preferred_language,buddy_language,category_id,is_translation)
            await sync_to_async(TranslationSession.objects.create)(
                preferred_language=preferred_language,
                buddy_language=buddy_language,
                original_text=transcription,
                translated_text=answer[0],
                suggested_response=answer[1],
                category_id=1
            )
            await self.send(text_data=json.dumps({
                        'translation': answer[0],
                        'suggestion': answer[1], # Just an example
                    }))
        elif action == 'get_data':
            # Fetch all data from the database
            saved_sessions = await sync_to_async(list)(
                TranslationSession.objects.all().values(
                    'id', 'preferred_language', 'buddy_language', 
                    'original_text', 'translated_text', 'suggested_response', 
                    'category_id', 'created_at'
                )
            )
            # Send the data back to the frontend
            await self.send(text_data=json.dumps({
                'message': 'Data retrieved successfully!',
                'data': saved_sessions
            }))

def process_audio_stream(audio_buffer):
    recognizer = sr.Recognizer()
    temp_audio_path = None

    try:
        # Convert audio buffer to PCM WAV if needed
        temp_audio_path = convert_to_wav(audio_buffer)

        # Process the converted PCM WAV file
        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)

        # Perform speech recognition
        text = recognizer.recognize_google(audio_data, language="de-DE")
        print(f"Real-time transcription: {text}")
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError as e:
        print(f"Recognition error: {e}")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Clean up the temporary file if it exists
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

def convert_to_wav(input_buffer):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_input:
        temp_input.write(input_buffer.read())
        temp_input_path = temp_input.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
        temp_output_path = temp_output.name
    
    try:
        subprocess.run([
            "ffmpeg", "-i", temp_input_path, 
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", temp_output_path
        ], check=True)
        return temp_output_path
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    finally:
        os.remove(temp_input_path)