import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("title", help="The title of the song")
parser.add_argument("artist", help="The name of the artist")
parser.add_argument("summarizer", help="The model to select the lyric. Choose from 'luhn' or 'lsa'. (Default: 'luhn')", nargs='?')
args = parser.parse_args()

if not args.summarizer:
    args.summarizer = 'luhn'

if __name__ == '__main__':
    song = get_lyrics(args)
    if song:
        lyrics = process_lyrics(song.lyrics)
        song_title, artist, summarizer = process_args(args)
        line = extract_lyric(lyrics, summarizer)
        prompt = generate_prompt(line, song_title, artist)
        img = generate_image(prompt)
        annotated_img = annotate(img, line)
        save_fig(annotated_img, song_title, artist, summarizer)

    else:
        print("There's been a problem retrieving lyrics. Please try again.")


# TODO: debug maximum line length
# TODO: remove contents of brackets from lyrics
# TODO: use textbbox or textlength instead of textsize
# TODO: validate args

