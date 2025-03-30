# pip install pyaudio
# pip install --upgrade google-cloud-speech

import pyaudio
import wave
import audioop
import io

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100  # sample rate
CHUNK = 1024  # frames per buffer

SILENCE_THRESHOLD = 500  # RMS value
SILENCE_DURATION = 2.0  # Duration of silence in seconds before stopping
MAX_RECORD_SECONDS = 30  # Optional: Maximum recording time in seconds

def record_voice():

    audio = pyaudio.PyAudio()
    
    # start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    
    frames = []
    silent_chunks = 0
    total_chunks = 0
    
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = audioop.rms(data, 2)  # get the RMS of the audio chunk
    
        if rms < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks = 0
    
        total_chunks += 1
    
        silence_time = silent_chunks * CHUNK / RATE
        total_time = total_chunks * CHUNK / RATE
    
        # Stop recording when silence duration exceeds SILENCE_DURATION or total record time exceeds MAX_RECORD_SECONDS
        if silence_time > SILENCE_DURATION or total_time > MAX_RECORD_SECONDS:
            break
    
    print("Finished recording")
    
    # stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Create a BytesIO object to hold the waveform in memory
    buffer = io.BytesIO()

    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return buffer
