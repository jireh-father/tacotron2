# -*- coding: utf-8 -*-
import os
import glob
import re

dirs = ["/home/ilseo/dataset/zeroth-korean/train_data_01/003", "/home/ilseo/dataset/zeroth-korean/test_data_01/003"]
output_path = "zeroth_files_and_scripts.txt"
output_file = open(output_path, "w+")
audio_dirs = []

for d in dirs:
    audio_dirs += glob.glob(os.path.join(d, "*"))
for audio_dir in audio_dirs:
    script_file = glob.glob(os.path.join(audio_dir, "*.txt"))[0]

    f = open(script_file, encoding="utf-8")
    lines = f.read().splitlines()
    f.close()
    for line in lines:
        line_split = line.split(" ")
        file_name = line_split[0]
        script = " ".join(line_split[1:])
        output_file.write(os.path.join(audio_dir, file_name + ".flac") + "|" + script + "\n")
output_file.close()