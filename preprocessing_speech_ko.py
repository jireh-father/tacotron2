# -*- coding: utf-8 -*-
import os
import glob
import re

data_dir = "/home/ilseo/dataset/speech_ko"
script_file = os.path.join(data_dir, "script_easy.txt")

f = open(script_file, encoding="utf-16")
lines = f.read().splitlines()
script_map = {}
cur_book_idx = 1
for line in lines:
    if not line:
        continue
    line = line.strip()
    if line[0] == '<':
        b_idx = line.split(".")[0][1:]
        if len(b_idx) == 1:
            b_idx = "0" + b_idx
        script_map[b_idx] = {}
        cur_book_idx = b_idx
    else:
        line_split = line.split(".")
        text_idx = line_split[0]
        if len(text_idx) == 1:
            text_idx = "0" + text_idx
        text = ".".join(line_split[1:]).strip()
        script_map[cur_book_idx][text_idx] = text

output_path = "speech_ko_files_and_scripts.txt"
output_file = open(output_path, "w+")
audio_dirs = glob.glob(os.path.join(data_dir, "*"))
for audio_dir in audio_dirs:
    if not os.path.isdir(audio_dir):
        continue
    wave_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    for wave_file in wave_files:
        fname = os.path.basename(wave_file)
        gr = re.search("^[a-z]{2}[0-9]{2}_t([0-9]{2})_s([0-9]{2})[.]wav$", fname)
        if not gr:
            continue
        b_idx = gr.group(1)
        t_idx = gr.group(2)
        script = script_map[b_idx][t_idx]
        output_file.write(wave_file + "|" + script + "\n")
output_file.close()