import os
from google.cloud import speech

# Path to the service account key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sodium-ray-397819-ac47b653e58b.json"

def speech_to_text(buffer):
    """Transcribes the audio file."""
    client = speech.SpeechClient()

    content = buffer.getvalue()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        language_code='fr-CA'  # Adjust this to the language of the audio
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print('Transcript: {}'.format(result.alternatives[0].transcript))

    return response
