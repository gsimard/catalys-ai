# pip install google-cloud-texttospeech
# pip install simpleaudio

import os
import io
import wave
import simpleaudio as sa
from google.cloud import texttospeech

# Path to the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sodium-ray-397819-ac47b653e58b.json"

def text_to_speech(input_text):
    """Converts input_text to speech, saving the output to output_file_name."""
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=input_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="fr-CA", name='fr-CA-Neural2-C'
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # The response's audio_content is the audio in byte format.
    audio_buffer = io.BytesIO(response.audio_content)

    # Use simpleaudio to play the audio
    with wave.open(audio_buffer, 'rb') as wave_read:
        wave_obj = sa.WaveObject.from_wave_read(wave_read)
        play_obj = wave_obj.play()
        play_obj.wait_done()
