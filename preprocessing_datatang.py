# -*- coding: utf-8 -*-
import os
import glob
import re

data_dir = "/home/ilseo/dataset/datatang"
output_path = "datatang_files_and_scripts.txt"
output_file = open(output_path, "w+")
audio_dirs = glob.glob(os.path.join(data_dir, "*"))
for audio_dir in audio_dirs:
    wave_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    for wave_file in wave_files:
        script_file = os.path.splitext(wave_file)[0] + ".txt"
        f = open(script_file, encoding="utf-8")
        lines = f.read().splitlines()
        f.close()
        script = lines[0]
        output_file.write(wave_file + "|" + script + "\n")
output_file.close()