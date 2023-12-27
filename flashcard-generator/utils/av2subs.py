import os
import math
import random
import subprocess

from pydub import AudioSegment
from pydub.silence import detect_silence
import pysrt

RUN_ID = random.randint(100000, 999999)

def audio2subs(input_file, whisper_model='small', temp_directory='temp', run_id=RUN_ID, verbose=False):  
    if verbose:
        print('\nAV2SUBS\n=======', flush=True)
        print(f'Run ID: {run_id}')
        print(f'Loading input file: {input_file}', flush=True)
    audio = loadAudio(input_file)
    
    if not os.path.exists(temp_directory):
        os.mkdir(temp_directory)
    
    if verbose:
        print('Breaking audio into chunks', flush=True)
    audio_chunks = chunkAudio(audio, temp_directory=temp_directory, run_id=run_id, verbose=verbose)

    if verbose:
        print('Feeding chunks into Whisper', flush=True)
    temp_srt_files = transcribeChunks(audio_chunks, whisper_model, temp_directory=temp_directory, run_id=run_id, verbose=verbose)

    if verbose:
        print('Combining output .srt files into one file', flush=True)
    combined_srt_file = combineSubtitleFiles(temp_srt_files, temp_directory, run_id=run_id)

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
# Whisper API has an input size limit of 25mb (although we currently don't need 
#  this since locally ran Whisper models don't have this restriction.)
def chunkAudio(audio, chunk_max_size_mb=25, min_silence_len=300, temp_directory='temp', run_id=RUN_ID, verbose=False):
    # We need to estimate the size of the audio in .wav to know how many chunks to make
    estimated_wav_size_bytes = (audio.frame_rate * audio.sample_width * audio.channels * (len(audio) / 1000))
    estimated_wav_size_mb = estimated_wav_size_bytes / (1024 * 1024)

    # Check if file size is already under max chunk size.
    if estimated_wav_size_mb < chunk_max_size_mb:
        return [audio], [[0, 0]]
    
    # Determine how many milliseconds of audio = chunk_max_size_mb
    # Multiple estimated_wav_size_mb by 110% to account for inaccuracies in estimate.
    chunk_max_len_ms = (len(audio) * chunk_max_size_mb) / (estimated_wav_size_mb * 1.10)

    if verbose:
        print('\tAnalyzing silences (this may take a few minutes)', flush=True)
    silence_ranges = detect_silence(audio, min_silence_len=min_silence_len, seek_step=1)

    chunks_saved = 0
    last_chunk_final_silence = [0, 0]
    output_files = []
    for i in range(1, len(silence_ranges)):
        current_silence_range = silence_ranges[i]
        last_silence_range = silence_ranges[i-1]
        
        # If this chunk is greater in size than 25mb, 
        #  add to audio_chunks and move onto the next chunk
        if current_silence_range[1] - last_chunk_final_silence[0] > chunk_max_len_ms:
            # We want to begin the chunk with the same silence that ended last chunk.
            # We want to end the chunk with silence as well.
            # We use last_silence_range to make sure this chunk is under chunk_max_size_mb
            audio_chunk = audio[last_chunk_final_silence[0]: last_silence_range[1]]
            chunks_saved += 1
            last_chunk_final_silence = last_silence_range
            output_files.append(f'{temp_directory}/{run_id}_chunked_audio_{chunks_saved}.wav')
            with open(output_files[-1], 'wb+') as out:
                audio_chunk.export(out, format='wav')
    
    # There will always be some extra audio not yet put into a chunk.
    audio_chunk = audio[last_chunk_final_silence[0]:]
    output_files.append(f'{temp_directory}/{run_id}_chunked_audio_{chunks_saved}.wav')
    with open(output_files[-1], 'wb+') as out:
        audio_chunk.export(out, format='wav')

    return output_files

# Use command-line Whisper to transcribe each chunk
def transcribeChunks(input_files, whisper_model, temp_directory='temp', run_id=RUN_ID, verbose=False):    
    output_files = []
    for i in range(len(input_files)):
        output_files.append(f'{temp_directory}/{run_id}_chunked_audio_{i+1}.srt')

        if os.path.exists(output_files[i]):
            if verbose:
                print(f'\tFound transcription for chunk {i+1}, skipping to next chunk. ({output_files[i]})')
            continue
        
        if verbose:
            print(f'\tWorking on transcribing chunk {i+1} of {len(input_files)}', flush=True)
        cmd = (f'whisper {input_files[i]} --model {whisper_model} '
               f'--output_dir {temp_directory} --output_format srt ' 
               f'--language zh --task transcribe --verbose False')
        subprocess.run(cmd, check=True)

    return [f'{temp_directory}/{run_id}_chunked_audio_{i+1}.srt' for i in range(len(input_files))]

# Combine the output .srt files
def combineSubtitleFiles(input_files, temp_directory='temp', run_id=RUN_ID):
    output_file_name = f'{temp_directory}/{run_id}_transcript.srt'
    if len(input_files) == 1:
        os.rename(input_files[0], output_file_name)
        return 0

    output_subs = pysrt.SubRipFile()
    subtitle_index = 1 # .srt files start with index 1
    buffer = pysrt.SubRipTime(0, 0, 0, 0)
    for file_index in range(len(input_files)):
        current_file = input_files[file_index]
        current_subs = pysrt.open(current_file)

        # We don't have to worry about the overlapping / duplicate silences introduced earlier.
        # For example, for chunks 1 and 2, the silence at the end of chunk 1 should NOT be reflected
        # within the subtitle timings, whereas that same silence WILL be reflected at the start of chunk 2.
        for sub in current_subs:
            new_sub = pysrt.SubRipItem(subtitle_index, 
                                       start=sub.start + buffer,
                                       end=sub.end + buffer, 
                                       text=sub.text)
            output_subs.append(new_sub)
            subtitle_index += 1
        
        buffer += current_subs[-1].end
        print(f'Buffer {file_index + 1}: {buffer}')
        
    output_subs.save(output_file_name, encoding='utf-8')

def combineTextFiles(input_files, output_file):
    out = open(output_file, 'w+', encoding='utf-8')
    for input_file in input_files:
        input_fp = open(input_file, 'r', encoding='utf-8')
        out.write(input_fp.readlines())

    out.close()

if __name__ == '__main__':
    input_file = 'test_file.mp4'
    whisper_model = 'large'

    audio2subs(input_file, whisper_model, run_id=RUN_ID, verbose=True)