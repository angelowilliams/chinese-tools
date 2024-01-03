import os
import random
import subprocess

from pydub import AudioSegment
import torch
import pysrt
import onnxruntime as ort

RUN_ID = random.randint(100000, 999999)
DEFAULT_TEMP_DIRECTORY = 'temp'
# silero-vad spits out tons of warnings that are out of our control without this
ort.set_default_logger_severity(3)
# silero-vad only suppored 16kHz and 8kHz sample rates
AUDIO_SAMPLE_RATE = 16000

def audio2subs(input_file, speech_threshold=0.5, whisper_model='small', whisper_prompt='以下为中文。', temp_directory=DEFAULT_TEMP_DIRECTORY, run_id=RUN_ID, verbose=False):  
    if verbose:
        print('\nAV2SUBS\n=======', flush=True)
        print(f'Run ID: {run_id}', flush=True)
    
    if not os.path.exists(temp_directory):
        os.mkdir(temp_directory)
    if not os.path.exists(f'{temp_directory}/{run_id}'):
        os.mkdir(f'{temp_directory}/{run_id}')

    if verbose:
        print(f'Loading input file: {input_file}', flush=True)
    audio, audio_path = extractAudio(input_file, temp_directory=temp_directory, run_id=run_id, verbose=verbose)
        
    if verbose:
        print('Breaking audio into chunks', flush=True)
    audio_chunks, speech_timestamps_ms = chunkAudio(audio, audio_path, threshold=speech_threshold, temp_directory=temp_directory, run_id=run_id, verbose=verbose)

    if verbose:
        print('Feeding chunks into Whisper', flush=True)
    temp_srt_files = transcribeChunks(audio_chunks, whisper_model, whisper_prompt, temp_directory=temp_directory, run_id=run_id, verbose=verbose)

    if verbose:
        print('Combining output .srt files into one file', flush=True)
    combined_srt_file = combineSubtitleFiles(temp_srt_files, speech_timestamps_ms, temp_directory=temp_directory, run_id=run_id)

    return combined_srt_file, temp_srt_files

# Extracts audio from file
def extractAudio(input_file, temp_directory=DEFAULT_TEMP_DIRECTORY, run_id=RUN_ID, verbose=False):
    file_type = input_file.split('.')[-1]
    try:
        audio = AudioSegment.from_file(input_file, file_type)
    except:
        raise Exception(f'File type {file_type} not supported.')
    
    audio = audio.set_frame_rate(AUDIO_SAMPLE_RATE)

    audio_path = f'{temp_directory}/{run_id}/audio.wav'
    if verbose:
        print(f'\tSaving extracted audio to {audio_path}')
    with open(audio_path, 'wb+') as out:
        audio.export(out, format='wav')
        
    return audio, audio_path

'''
chunkAudio breaks the input audio into chunks such that there are no silences
longer than min_silence_len milliseconds. This is imporant for two reasons:
1. Whisper tends to hallucinate / breakdown during silences
2. If you wanted to use Whisper API (which I don't do here), 
   you'll need to make sure each chunk is <25mb

Potential improvements:
1. Use voice activity detection to remove chunks that have sound (ie. not
   silent) but no words spoken.
2. Make sure that no chunk is greater than 25mb. (See also: old chunkAudio
   audio version at the end of this file)
3. We could try to isolate the voices from any backgroud noise.
'''
def chunkAudio(audio, audio_path, threshold=0.5, temp_directory=DEFAULT_TEMP_DIRECTORY, run_id=RUN_ID, verbose=False):
    output_dir = f'{temp_directory}/{run_id}/audio_chunks'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if verbose:
        print('\tLoading silero-vad and detecting speech timestamps', flush=True)
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                model='silero_vad',
                                onnx=True,
                                verbose=False)
    (get_speech_timestamps, _, read_audio, _, _) = utils
    audio_for_vad = read_audio(audio_path, AUDIO_SAMPLE_RATE)
    speech_timestamps = get_speech_timestamps(audio_for_vad, 
                                            model, 
                                            threshold=threshold,
                                            sampling_rate=AUDIO_SAMPLE_RATE,
                                            speech_pad_ms=200,
                                            min_silence_duration_ms=200)
    
    if verbose:
        print(f'\tSaving audio chunks', flush=True)
    output_files = []
    speech_timestamps_ms = []
    samples_to_ms_factor = AUDIO_SAMPLE_RATE / 1000
    for i in range(len(speech_timestamps)):
        speech_start_ms = speech_timestamps[i]['start'] // samples_to_ms_factor
        speech_end_ms = speech_timestamps[i]['end'] // samples_to_ms_factor
        speech_timestamps_ms.append((speech_start_ms, speech_end_ms))

        audio_chunk = audio[speech_start_ms: speech_end_ms]
        output_files.append(f'{output_dir}/chunked_audio_{i+1}.wav')
        with open(output_files[-1], 'wb+') as out:
            audio_chunk.export(out, format='wav')

    return output_files, speech_timestamps_ms

# Use command-line Whisper to transcribe each chunk
def transcribeChunks(input_files, whisper_model, whisper_prompt, temp_directory=DEFAULT_TEMP_DIRECTORY, run_id=RUN_ID, verbose=False):    
    output_dir = f'{temp_directory}/{run_id}/transcript_chunks'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_files = []
    for i in range(len(input_files)):
        output_files.append(f'{output_dir}/chunked_audio_{i+1}.srt')
        
        if verbose:
            print(f'\tWorking on transcribing chunk {i+1} of {len(input_files)}', flush=True)
        cmd = (f'whisper {input_files[i]} --model {whisper_model} '
               f'--initial_prompt "{whisper_prompt}" --patience 2 '
               f'--output_dir {output_dir} --output_format srt ' 
               f'--language zh --task transcribe --verbose False')
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    return output_files

# Combine the output .srt files
def combineSubtitleFiles(input_files, speech_timestamps, temp_directory=DEFAULT_TEMP_DIRECTORY, run_id=RUN_ID):
    output_file_name = f'{temp_directory}/{run_id}/transcript.srt'
    if len(input_files) == 1:
        os.rename(input_files[0], output_file_name)
        return 0

    output_subs = pysrt.SubRipFile()
    subtitle_index = 1 # .srt files start with index 1
    for chunk_index in range(len(input_files)):
        current_file = input_files[chunk_index]
        current_subs = pysrt.open(current_file)

        # If the .srt file only has one sub, we'll just use the timestamp from silero_vad
        if len(current_subs) == 1:
            output_subs.append(pysrt.SubRipItem(subtitle_index, 
                            start=pysrt.SubRipTime(milliseconds=speech_timestamps[chunk_index][0]),
                            end=pysrt.SubRipTime(milliseconds=speech_timestamps[chunk_index][1]),
                            text=current_subs[0].text))
            subtitle_index += 1
        # If more than one sub, we'll use the VAD timestamp as the start of the first sub
        #   and the end for the final sub, then use the Whisper .srt timestamps for all
        #   the other subtitles timings. 
        else:
            chunk_start = pysrt.SubRipTime(milliseconds=speech_timestamps[chunk_index][0])
            chunk_end = pysrt.SubRipTime(milliseconds=speech_timestamps[chunk_index][1])
            for sub_i in range(len(current_subs)):
                if sub_i == 0:
                    # First sub in this file
                    start = chunk_start
                    end = chunk_start + current_subs[sub_i].end
                elif sub_i == len(current_subs) - 1:
                    # Last sub in this file
                    start = chunk_start + current_subs[sub_i].start
                    end = chunk_end
                else:
                    start = chunk_start + current_subs[sub_i].start
                    end = chunk_start + current_subs[sub_i].end
                
                output_subs.append(pysrt.SubRipItem(subtitle_index, 
                                start=start,
                                end=end,
                                text=current_subs[sub_i].text))
                subtitle_index += 1
        
    output_subs.save(output_file_name, encoding='utf-8')

if __name__ == '__main__':
    input_file = 'qiaohuDVD01.mp4'
    whisper_model = 'large-v2'
    whisper_prompt = '以下为中文。'
    threshold = 0.7
    #run_id = 497139
    #temp_srt_files = [f'temp/{run_id}_chunked_audio_{i+1}.srt' for i in range(21)]
    #combineSubtitleFiles(temp_srt_files, run_id=run_id)
    audio2subs(input_file, 
               speech_threshold=threshold, 
               whisper_model=whisper_model, 
               whisper_prompt=whisper_prompt, 
               run_id=RUN_ID, 
               verbose=True)


# This is an older version of chunkAudio that had the functionality to divide up
# the audio such that no chunk was larger than 25mb. I removed this functionality
# for the sake of simplicity, since I'm only using locally-ran Whisper right now
# anyway.
# Saving this code down here for whenever I chose to reintroduce that functionality.
'''
def chunkAudio(audio, chunk_max_size_mb=25, min_silence_len=300, temp_directory=DEFAULT_TEMP_DIRECTORY, run_id=RUN_ID, verbose=False):
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
            print(f'Adding file: {temp_directory}/{run_id}_chunked_audio_{chunks_saved}.wav', flush=True)
            output_files.append(f'{temp_directory}/{run_id}_chunked_audio_{chunks_saved}.wav')
            with open(output_files[-1], 'wb+') as out:
                audio_chunk.export(out, format='wav')
    
    # There will always be some extra audio not yet put into a chunk.
    audio_chunk = audio[last_chunk_final_silence[0]:]
    print(f'Adding extra audio file: {temp_directory}/{run_id}_chunked_audio_{chunks_saved+1}.wav', flush=True)
    output_files.append(f'{temp_directory}/{run_id}_chunked_audio_{chunks_saved+1}.wav')
    with open(output_files[-1], 'wb+') as out:
        audio_chunk.export(out, format='wav')

    return output_files
'''