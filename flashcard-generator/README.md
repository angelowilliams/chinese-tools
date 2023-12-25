# flashcard-generator
This script combines various utilities to create Pleco flashcards out of Chinese media.
The flashcards are sequenced such that they incrementally introduce vocabulary wihtin the media.

## Usage
`python flashcard-generator.py ebook.epub video.mp4 audio.mp3 --known_word_list known_words.txt --output

TODO: Currently, only `.mp4`, `.mp3`, and `.epub` files are supported.

## Utilities
This script combines a few tools, found in the `utils/` directory. These tools are:
- `txt2pleco.py`: Given a file of line-separated sequential sentences, create Pleco flashcards.
- `video2audio.py`: Given an `.mp4` file, create a `.mp3` file.
- `audio2txt.py`: Given an `.mp3` file, uses OpenAI's Whisper language model to create a transcribed text file of sequential sentences.
- `epub2txt.py`: Given an `.epub` file, create a text file of sequential sentences.

## Dependencies
Python version must be `>=3.8` but `<=3.11`.

Other dependencies depends upon what type(s) of files are input.

### Video
* [`moviepy`](https://github.com/Zulko/moviepy/tree/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e)
    - When newer versions of `moviepy`, you may encounter an issue
* [`whisper`](https://github.com/openai/whisper)
    - If applicable, make sure that your `torch` installation is setup for CUDA.

### Audio
TODO

### Ebook
TODO

## Importing flashcards into Pleco
TODO