import pysrt

LATIN_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
CHARS_TO_IGNORE = ',.?! 。…'
# Whisper tends to hallucinate a specific set of tokens. These are the ones
# I've found so far:
BLACKLISTED_LINES = ['未经许可,不得翻唱或使用',
                     '请不吝点赞 订阅 转发 打赏支持明镜与点点栏目']

def subs2txt(input_file, output_file, min_chars=5, no_english=True, verbose=False):
    if verbose:
        print('\nSUBS2TXT\n=======', flush=True)
    
    output = open(output_file, 'wb+')
    subs = pysrt.open(input_file)
    texts_to_write = []
    for sub in subs:
        # Check if subtitle isn't long enough
        qualifying_chars = [char in sub.text for char in sub.text if char not in CHARS_TO_IGNORE]
        if len(qualifying_chars) < min_chars:
            continue

        # If applicable, check if subtitle contains any latin letters.
        if no_english:
            if any([char in LATIN_ALPHABET for char in sub.text]):
                continue
        
        if sub.text in BLACKLISTED_LINES:
            continue
        
        texts_to_write.append(sub.text)
    
    # Remove any duplicate lines
    texts_to_write = list(set(texts_to_write))

    output.write('\n'.join(texts_to_write).encode('utf-8'))
    output.close()

if __name__ == '__main__':
    input_file = 'transcript.srt'
    output_file = 'transcript.txt'
    subs2txt(input_file, output_file)        
