import subprocess
from record_voice import record_voice
from speech_to_text import speech_to_text
from gpt import gpt_request
from text_to_speech import text_to_speech

buffer = record_voice()
prompt = speech_to_text(buffer)
response = gpt_request(prompt)
text_to_speech(response)
