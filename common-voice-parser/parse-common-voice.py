import os
import shutil
from mutagen.id3 import ID3NoHeaderError, ID3, COMM

'''
We expect other.tsv in common-voice to have the following schema:
client_id	path	sentence	up_votes	down_votes	age	gender	accents	variant	locale	segment
25bc975d06200b7b1c9135db090561cb0d9b28d172e51c6826f9e665a3ba7bddda6ec0c6c09102f5d1d4ec99cd094b9c91ad0141e49341246bc6eb2e7167f2e7	common_voice_zh-CN_19703883.mp3	模式种采样自台湾龟山岛。	1	0	thirties	female	出生地：31 上海市		zh-CN	
25bc975d06200b7b1c9135db090561cb0d9b28d172e51c6826f9e665a3ba7bddda6ec0c6c09102f5d1d4ec99cd094b9c91ad0141e49341246bc6eb2e7167f2e7	common_voice_zh-CN_19706151.mp3	后者娶天之瓮主神。	1	0	thirties	female	出生地：31 上海市		zh-CN	
faa0ecc626e80638016d6295b5018372b8567f5f3177f68039e8049b12b555163027f6ebdeb13fd1771de64bb828ff4ff7c28d3258976a05da6b8baad061aae1	common_voice_zh-CN_19961025.mp3	贝尔卢。	1	0	twenties	male	出生地：32 江苏省		zh-CN	
'''

CV_CLIPS_FOLDER = 'cv-corpus/zh-CN/clips'
CV_INDEX = 'cv-corpus/zh-CN/validated.tsv'
NEW_AUDIO_FOLDER = 'audio/to_do'
PRACTICED_AUDIO_FOLDER = 'audio/complete'

import sys
sys.stdout.reconfigure(encoding='utf-8')

def parse_common_voice(cv_index, cv_clips, new_audio_folder, practiced_audio_folder, known_words, min_upvotes=2, genders=['male', ''], min_sentence_len=5):
    index_file = open(cv_index, 'r', encoding='utf-8')
    lines = index_file.readlines()
    for line in lines[1:]:
        values = line.split('\t')
        clip_name = values[1]
        sentence = values[2]
        upvotes = values[3]
        gender = values[6]

        # Check if sentence meets our acceptance criteria
        if gender not in genders or \
           int(upvotes) < min_upvotes or \
           len(sentence) < min_sentence_len or \
           any([character not in known_words for character in sentence]):
            continue
    
        # Check if clip has already been parsed
        if os.path.exists(new_audio_folder + '/' + clip_name) or \
           os.path.exists(practiced_audio_folder + '/' + clip_name):
            continue
        
        print(f'Added {clip_name}! Sentence is 『 {sentence} 』', flush=True)
        # Copy clip to new_audio_folder
        output_file = new_audio_folder + '/' + clip_name
        shutil.copy(cv_clips + '/' + clip_name, output_file)

        # Adding lyrics
        try: 
            tags = ID3(output_file)
        except ID3NoHeaderError:
            print("Adding ID3 header")
            tags = ID3()
        
        tags['COMM'] = COMM(encoding=3, lang=u'chi', desc='desc', text=sentence)
        tags.save(output_file)

    index_file.close()

if __name__ == '__main__':
    known_words_file = '../known_words_01_02_24.txt'
    known_words = []
    with open(known_words_file, 'r', encoding='utf-8') as fp:
        for word in fp.readlines():
            known_words.append(word.replace('\n', ''))

    parse_common_voice(CV_INDEX, CV_CLIPS_FOLDER, NEW_AUDIO_FOLDER, PRACTICED_AUDIO_FOLDER, known_words)