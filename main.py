import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("title", help="The title of the song")
parser.add_argument("artist", help="The name of the artist")
parser.add_argument("summarizer", help="The model to select the lyric. Choose from 'luhn' or 'lsa'. (Default: 'luhn')", nargs='?')
parser.add_argument("--magic_prompt", help="use prompt engineer model?", action='store_true', default=False)

args = parser.parse_args()

if not args.summarizer:
    args.summarizer = 'luhn'


def main():
    print('args:', args)
    song_title, artist, summarizer, magic_prompt = process_args(args)
    song = get_lyrics(song_title, artist)

    if song:
        lyrics = process_lyrics(song.lyrics)
        line = extract_lyric(magic_prompt, lyrics, summarizer)
        prompt = generate_prompt(magic_prompt, line, song_title, artist)
        print('image prompt: ', prompt)
        img = generate_image(prompt)

        if img:
            annotated_img = annotate(img, line.lower())
            save_fig(annotated_img, song_title, artist, summarizer, magic_prompt)
        else:
            print("There's been a problem generating the image. Please try again.")

    else:
        print("There's been a problem retrieving lyrics. Did you spell correctly?")


if __name__ == '__main__':
    main()


# TODO: use textbbox or textlength instead of textsize

