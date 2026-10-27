# VLM images and crops at most 1024 px on the long side; 1568 doubled the image tokens on
# danbooru's 4 Mpx images with no better graphs. Layer: ... configs/controls/side1024.py

controls = {"max_side": 1024}
