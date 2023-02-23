from utils import *
from PIL import Image
import textwrap


def test_annotate(img, caption):
    border_size = img.size[0] // 10
    font_size = border_size // 3
    img_with_border = ImageOps.expand(img, border=border_size, fill='black')

    draw = ImageDraw.Draw(img_with_border)
    font = ImageFont.truetype("fonts/Courier Prime Bold.ttf", font_size)
    text_size = draw.textsize(caption, font=font)

    x = (img_with_border.width - text_size[0]) / 2
    y = img_with_border.height - (border_size / 2) - text_size[1] + 7

    if text_size[0] > img_with_border.width:
        midpoint = len(caption) // 2

        for i, line in enumerate(textwrap.wrap(caption, width=midpoint+5)):
            line_text_size = draw.textsize(line, font=font)
            line_x = (img_with_border.width - line_text_size[0]) / 2
            line_y = y + (font_size * i) + 7

            draw.text((line_x, line_y), line, fill='red', font=font)

        img_with_border = ImageOps.expand(img_with_border, border=int(border_size * 0.5), fill='black')
    else:
        draw.text((x, y), caption, fill='red', font=font)

    return img_with_border


song_title = 'october sky'
artist = 'yebba'
summarizer = 'lsa'

song = get_lyrics(song_title, artist)
lyrics = process_lyrics(song.lyrics)
line = extract_lyric(lyrics, summarizer)

img = Image.open('examples/test_image_yebba.png')
annotated_image = test_annotate(img, line)

save_fig(annotated_image, 'test_'+song_title, artist, summarizer, test=True)

