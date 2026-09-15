"""Build the .docx from a height-capped copy of the reference HTML.

Word truncates an inline image taller than the text body rather than scaling
it: four of the seven flow charts came through cut off. pandoc takes the image
width from the HTML's max-width and lets height follow the aspect ratio, so the
fix is to cap the width of the tall charts for the Word pass only. The HTML and
PDF keep the wider rendering, where a tall figure simply flows onto its own
page.
"""
import base64, io, re, subprocess, sys
from PIL import Image

SRC, TMP, OUT = 'control_law_reference.html', '_for_docx.html', 'control_law_reference.docx'
MAX_H_IN = 8.2          # Word text body is ~9 in; leave margin for the caption
DPI = 96                # what pandoc assumes for CSS px

s = open(SRC).read()
imgs = list(re.finditer(
    r'<img style="max-width:(\d+)px" src="data:image/png;base64,([A-Za-z0-9+/=]+)"', s))
# pandoc ignores max-width entirely -- it honours an explicit width, so the
# docx copy carries width= on every chart.
print(f'{len(imgs)} flow charts')

out, last, n = [], 0, 0
for m in imgs:
    cur_w = int(m.group(1))
    im = Image.open(io.BytesIO(base64.b64decode(m.group(2))))
    ar = im.size[0] / im.size[1]
    cap_w = int(MAX_H_IN * DPI * ar)
    new_w = min(cur_w, cap_w)
    h_in = new_w / DPI / ar
    if new_w != cur_w:
        n += 1
        print(f'   {im.size}  ar {ar:.3f}   {cur_w}px -> {new_w}px '
              f'(height {cur_w/DPI/ar:.2f} -> {h_in:.2f} in)')
    out.append(s[last:m.start()])
    out.append(f'<img width="{new_w}" style="width:{new_w}px" '
               f'src="data:image/png;base64,{m.group(2)}"')
    last = m.end()
out.append(s[last:])
open(TMP, 'w').write(''.join(out))
print(f'{n} narrowed for Word')
subprocess.run(['pandoc', TMP, '-o', OUT], check=True)
print('wrote', OUT)
