"""Inline the model weights and the JS U-Net into index.html.

Writes heaviside_demo.html, a single self-contained page (no server needed;
open it in a browser). Run after train_demo.py:

    python demo/heaviside_demo/build_page.py
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def read(name):
    with open(os.path.join(HERE, name)) as f:
        return f.read()


if __name__ == '__main__':
    page = read('index.html')
    weights, unet = read('model_weights.js'), read('unet.js')
    assert '</script' not in weights and '</script' not in unet
    page = page.replace('<!-- INLINE:model_weights.js -->', '<script>\n' + weights + '</script>')
    page = page.replace('<!-- INLINE:unet.js -->', unet)
    assert '<!-- INLINE:' not in page
    out = os.path.join(HERE, 'heaviside_demo.html')
    with open(out, 'w') as f:
        f.write(page)
    print(f'wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB)')
