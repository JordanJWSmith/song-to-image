import os
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
    if 'You might also like' in lyrics:  # often webscraped erroneously
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
    for i in range(10):
        try:
            print('Generating image...')
            # endpoint_url = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
            endpoint_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"

            payload = {"inputs": prompt}
            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "application/json",
                "Accept": "image/png"
            }
            response = r.post(endpoint_url, headers=headers, json=payload)
            img = Image.open(BytesIO(response.content))

            return img
        except:
            continue
    return None


def annotate(img, caption):
    border_size = 50
    img_with_border = ImageOps.expand(img, border=border_size, fill='black')

    draw = ImageDraw.Draw(img_with_border)
    font = ImageFont.truetype("fonts/Courier Prime Bold.ttf", 16)
    text_size = draw.textsize(caption, font=font)

    x = (img_with_border.width - text_size[0]) / 2
    y = img_with_border.height - (border_size/2) - text_size[1] + 7
    draw.text((x, y), caption.lower(), fill='red', font=font)

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

