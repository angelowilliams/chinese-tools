import os
from datetime import timedelta

import whisper
import moviepy.editor as mp
import numpy as np
import torch

WHISPER_MODEL = 'tiny'
INPUT_FILE = os.path.abspath('test_file.mp4')
OUTPUT_SRT_FILE = 'test.srt'
OUTPUT_TXT_FILE = 'test.txt'
LANGUAGE = 'zh'

def main(input_file, output_srt_file, whisper_model, output_txt_file=None):
    print('Loading audio', flush=True)
    audio = loadAudio(input_file)[:5000]

    print('Getting transcript', flush=True)
    transcript = getTranscript(audio, whisper_model)

    print('Saving to .srt file', flush=True)
    saveTranscriptToSRT(transcript, output_srt_file)

    if output_txt_file:
        print('Saving to .txt file', flush=True)
        saveTranscriptToTXT(transcript, output_txt_file)

# loads audio track from input file
def loadAudio(file):
    video = mp.VideoFileClip(file)
    audio = video.audio
    audio.write_audiofile(INPUT_FILE[:-4] + "_audio.mp3")
    audio_array = audio.to_soundarray()

    video.close()
    return audio_array.astype(np.float32)

# using the audio track, uses whisper model to get transcription
def getTranscript(audio, whisper_model):
    torch.cuda.init()

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device selected for transcription: " + device.type, flush=True)

    model = whisper.load_model(whisper_model)
    model = model.to(device)
    
    return model.transcribe(audio, language=LANGUAGE, verbose=True, fp16=False)

# saves the transcript as an .srt file
# heavy inspiration drawn from: https://github.com/openai/whisper/discussions/98#discussioncomment-3725983
def saveTranscriptToSRT(transcript, output_file):
    srt_file = open(output_file, 'w+', encoding='utf-8')
    segments = transcript['segments']
    
    for segment in segments:
        start_time = str(0)+str(timedelta(seconds=int(segment['start'])))+',000'
        end_time = str(0)+str(timedelta(seconds=int(segment['end'])))+',000'
        text = segment['text']
        segmentId = segment['id']+1
        segment = f"{segmentId}\n{start_time} --> {end_time}\n{text[1:] if text[0] == ' ' else text}\n\n"

        srt_file.write(segment)

# saves the transcript as a .txt file
def saveTranscriptToTXT(transcript, output_file):
    txt_file = open(output_file, 'w+', encoding='utf-8')
    segments = transcript['segments']
    
    for segment in segments:
        text = segment['text']
        txt_file.write(text)

if __name__ == '__main__':
    main(INPUT_FILE, OUTPUT_SRT_FILE, WHISPER_MODEL, output_txt_file=OUTPUT_TXT_FILE)