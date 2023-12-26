import os
import math
import random
import subprocess

from pydub import AudioSegment
from pydub.silence import detect_silence
import pysrt

RUN_ID = random.randint(1000, 9999)

def audio2subs(input_file, output_srt_file, whisper_model, temp_directory='temp', run_id=RUN_ID, verbose=False):  
    if verbose:
        print('\nAV2SUBS\n==========', flush=True)
        print(f'Loading input file: {input_file}', flush=True)
    audio = loadAudio(input_file)

    if verbose:
        print('Breaking audio into chunks', flush=True)
    audio_chunks = chunkAudio(audio, input_file, verbose=verbose)

    if verbose:
        print('Feeding chunks into Whisper', flush=True)
    temp_srt_files = transcribeChunks(audio_chunks, whisper_model, temp_directory, verbose=verbose)

    if verbose:
        print('Combining output .srt files into one file', flush=True)
    combined_srt_file = combineSubtitleFiles(temp_srt_files, output_srt_file)

    return combined_srt_file, temp_srt_files

# Loads audio from file
def loadAudio(input_file):
    file_type = input_file.split('.')[-1]
    try:
        audio = AudioSegment.from_file(input_file, file_type)
    except:
        raise Exception(f'File type {file_type} not supported.')
    
    return audio

# Break audio into chunks
# Try to avoid splitting in the middle of sentences
# Whisper has an input size limit of 25mb, use default of 20mb in case silences are very sparse.
def chunkAudio(audio, input_file, chunk_max_size_mb=20, min_silence_len=3000, verbose=False):
    audio_size_mb = os.stat(input_file).st_size / (1024 * 1024)
    number_of_chunks = math.ceil(audio_size_mb / chunk_max_size_mb)
    if number_of_chunks == 1:
        # File size is already under max chunk size.
        return [audio]

    if verbose:
        print('\tAnalyzing silences (this may take a few minutes)', flush=True)
    silence_ranges = detect_silence(audio, min_silence_len=min_silence_len, seek_step=5)

    max_chunk_len = len(audio) / number_of_chunks
    audio_chunks = []
    last_chunk_start = 0
    for silence_range in silence_ranges:
        if silence_range[1] > last_chunk_start + max_chunk_len:
            audio_chunks.append(audio[last_chunk_start: silence_range[1]])
            last_chunk_start = silence_range[0]
            if verbose:
                print(f'\tFound end of chunk {len(audio_chunks)} of {number_of_chunks}', flush=True)
            if len(audio_chunks) == number_of_chunks:
                    break
    
    if len(audio_chunks) != number_of_chunks:
        audio_chunks.append(audio[last_chunk_start:])
        print(f'\tRemaining audio added to chunk {len(audio_chunks)}', flush=True)

    return audio_chunks

# Use command-line Whisper to transcribe each chunk
def transcribeChunks(audio_chunks, whisper_model, temp_directory='temp', run_id=RUN_ID, verbose=False):
    if not os.path.exists(temp_directory):
        os.mkdir(temp_directory)

    output_transcripts = []
    for i in range(len(audio_chunks)):
        if verbose:
            print(f'\tWorking on transcripting chunk {i+1} of {len(audio_chunks)}', flush=True)

        chunk = audio_chunks[i]
        chunk_file = f'{temp_directory}/{run_id}_chunked_audio_{i+1}.wav'
        with open(chunk_file, 'wb+') as out:
            chunk.export(out, format='wav')

        cmd = (f'whisper {chunk_file} --model {whisper_model} '
               f'--output_dir {temp_directory} --output_format srt ' 
               f'--language zh --task transcribe --verbose False')
        subprocess.run(cmd, check=True)

        output_transcripts.append(f'{temp_directory}/{run_id}_chunked_audio_{i+1}.srt')

    return output_transcripts

# Combine the output .srt files
def combineSubtitleFiles(input_files, output_file, silence_buffer=3000, run_id=RUN_ID):
    if len(input_files) == 1:
        os.rename(input_files[0], output_file)
        return 0

    output_subs = pysrt.open(input_files[0])
    final_sub = output_subs[-1]
    buffer = final_sub.end

    for file_index in range(1, len(input_files)):
        current_file = input_files[file_index]
        subs = pysrt.open(current_file)

        # We have to shift all subtitles after the first chunk to account for
        # the silence buffers added in chunkAudio
        subs.shift(seconds =- silence_buffer)

        for sub in subs:
            sub.start += buffer
            sub.end += buffer

        final_sub = output_subs[-1]
        buffer += final_sub.end
        
    output_subs.save(output_file, encoding='utf-8')

if __name__ == '__main__':
    input_file = 'test_file.mp4'
    output_srt_file = 'transcript.srt'
    whisper_model = 'tiny'

    audio2subs(input_file, output_srt_file, whisper_model, verbose=True)