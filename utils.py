import os
import re
import lyricsgenius
import requests as r
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from PIL import Image, ImageOps, ImageDraw, ImageFont

load_dotenv()

GENIUS_TOKEN = os.getenv('GENIUS_TOKEN')
HF_TOKEN = os.getenv('HF_TOKEN')

genius = lyricsgenius.Genius(GENIUS_TOKEN)
genius.remove_section_headers = True

summarizers = {
    'luhn': LuhnSummarizer(),
    'lsa': LsaSummarizer(),
    'lexrank': LexRankSummarizer()
}


def process_args(args):
    if args.summarizer not in summarizers.keys():
        raise KeyError(f'The specified summarizer is not available. Please choose from: {list(summarizers.keys())}')
    return args.title.title(), args.artist.title(), args.summarizer


def get_lyrics(title, artist):
    # Scraping the lyrics can sometimes time out.
    # Give it 10 tries before giving up.
    for i in range(10):
        try:
            song = genius.search_song(title, artist)
            return song
        except Exception:
            continue
    return None


def process_lyrics(text):
    lyrics = text.split('\n')[:-1]

    brackets = r'\s*\([^()]*\)'
    lyrics = [re.sub(brackets, '', line) for line in lyrics]  # remove contents of parentheses, i.e. "(whoa-oh)"

    if 'You might also like' in lyrics:  # often webscraped erroneously from Genius
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
    print('Generating image...')

    # endpoint_url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
    endpoint_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"

    payload = {
        "inputs": prompt
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "image/png"
    }
    response = r.post(endpoint_url, headers=headers, json=payload)
    img = Image.open(BytesIO(response.content))

    print('Done.')
    return img


def annotate(img, caption):
    border_size = img.size[0] // 10
    font_size = border_size // 3
    img_with_border = ImageOps.expand(img, border=border_size, fill='black')

    draw = ImageDraw.Draw(img_with_border)
    font = ImageFont.truetype("fonts/Courier Prime Bold.ttf", font_size)
    text_size = draw.textsize(caption, font=font)

    x = (img_with_border.width - text_size[0]) / 2
    y = img_with_border.height - (border_size/2) - text_size[1] + 7
    draw.text((x, y), caption.lower(), fill='red', font=font)

    return img_with_border


def save_fig(img, song_title, artist, summarizer):
    chars = r'[<>:"/\\|?*\s]'
    song_title = re.sub(chars, '_', song_title).lower()
    artist = artist.lower().replace(" ", "_")

    now = datetime.now()
    str_timestamp = now.strftime("%d-%m-%y-%H%M%S")

    save_dir = os.path.join('output', f'{song_title}_{artist}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = os.path.join(save_dir, f'{song_title}_{summarizer}_{str_timestamp}.png')

    img.save(save_path)

    print(f'Image saved at {save_path}')

