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
    song_title, artist, summarizer = process_args(args)
    song = get_lyrics(song_title, artist)

    if song:
        lyrics = process_lyrics(song.lyrics)
        line = extract_lyric(lyrics, summarizer)
        prompt = generate_prompt(line, song_title, artist)
        img = generate_image(prompt)

        if img:
            annotated_img = annotate(img, line)
            save_fig(annotated_img, song_title, artist, summarizer)
        else:
            print("There's been a problem generating the image. Please try again.")

    else:
        print("There's been a problem retrieving lyrics. Did you spell correctly?")


# TODO: debug maximum line length
# TODO: use textbbox or textlength instead of textsize

