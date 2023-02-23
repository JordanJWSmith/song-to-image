import os
import argparse
import lyricsgenius
import requests as r
from io import BytesIO
from datetime import datetime
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.parsers.plaintext import PlaintextParser
from PIL import Image, ImageOps, ImageDraw, ImageFont

summarizers = {
    'luhn': LuhnSummarizer(),
    'lsa': LsaSummarizer()
}

GENIUS_TOKEN = "o6J2ynO-NVhUFOsm_DKS5pVyzlq5DY6PmOxScdJ-EK_kdKu_Kapz-zJKzDTfEL9a"
HF_TOKEN = "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"


def get_lyrics(args):
    for i in range(10):
        try:
            song = genius.search_song(args.title, args.artist)
            return song
        except Exception:
            continue
    return None


def process_lyrics(text):
    lyrics = text.split('\n')[:-1]
    if 'You might also like' in lyrics:
        lyrics.remove('You might also like')
    lyrics = '\n'.join(lyrics)
    return lyrics


def extract_lyric(text, summarizer):
    lyrics = text.replace('\n', '. ').replace(' . ', ' ')
    my_parser = PlaintextParser.from_string(lyrics, Tokenizer('english'))
    summarizer_model = summarizers[summarizer]
    summary = summarizer_model(my_parser.document, sentences_count=1)

    return [str(sentence)[:-1] for sentence in summary][0]


def generate_prompt(text, title, artist):
    return f'{text}. {title} by {artist}. Oil painting, detailed.'


def generate_image(prompt):
    endpoint_url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"  # url of your endpoint

    payload = {"inputs": prompt}
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "image/png" # important to get an image back
    }
    response = r.post(endpoint_url, headers=headers, json=payload)
    print(response)
    img = Image.open(BytesIO(response.content))

    return img


def annotate(img, caption):
    border_size = 50
    img_with_border = ImageOps.expand(img, border=border_size, fill='black')

    draw = ImageDraw.Draw(img_with_border)
    font = ImageFont.truetype("fonts/Courier Prime Bold.ttf", 16)
    text_size = draw.textsize(caption, font=font)

    x = (img_with_border.width - text_size[0]) / 2
    y = img_with_border.height - (border_size/2) - text_size[1] + 7
    draw.text((x, y), caption, fill='red', font=font)

    return img_with_border


def save_fig(img, song_title, artist, summarizer):
    song_title = song_title.lower().replace(" ", "_")
    artist = artist.lower().replace(" ", "_")

    now = datetime.now()
    str_timestamp = now.strftime("%d-%m-%y-%H%M%S")

    save_dir = os.path.join('output', f'{song_title}_{artist}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img.save(os.path.join(save_dir, f'{song_title}_{summarizer}_{str_timestamp}.jpg'))


parser = argparse.ArgumentParser()
parser.add_argument("title", help="The title of the song")
parser.add_argument("artist", help="The name of the artist")
parser.add_argument("summarizer", help="The model to select the lyric. Choose from 'luhn' or 'lsa'. (Default: 'luhn')", nargs='?')
args = parser.parse_args()

if not args.summarizer:
    args.summarizer = 'luhn'

genius = lyricsgenius.Genius(GENIUS_TOKEN)
genius.remove_section_headers = True


song = get_lyrics(args)
if song:
    lyrics = process_lyrics(song.lyrics)
    summarizer = args.summarizer
    line = extract_lyric(lyrics, summarizer)
    song_title = args.title.title()
    artist = args.artist.title()
    prompt = generate_prompt(line, song_title, artist)
    img = generate_image(prompt)
    annotated_img = annotate(img, line)
    save_fig(annotated_img, song_title, artist, summarizer)

else:
    print("There's been a problem. Please try again.")


# TODO: debug maximum line length
# TODO: remove contents of brackets from lyrics
# TODO: use textbbox or textlength instead of textsize
# TODO: validate args

