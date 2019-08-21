# -*- coding: utf-8 -*-
import os
import glob
import re
import re
from scipy.io.wavfile import write
import numpy as np



output_path = "aihub_files_and_scripts.txt"
output_file = open(output_path, "w+")
wav_dir = "/home/ilseo/dataset/KsponSpeech_wav"
if not os.path.isdir(wav_dir):
    os.makedirs(wav_dir)
data_root = "/home/ilseo/dataset/KsponSpeech"
data_dirs = [d for d in glob.glob(data_root + "/*") if os.path.isdir(d)]
for data_dir in data_dirs:
    sub_data_dirs = [d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)]
    for sub_data_dir in sub_data_dirs:
        pcm_files = glob.glob(os.path.join(sub_data_dir, "*.pcm"))
        for pcm_file in pcm_files:
            data = np.memmap(pcm_file, dtype='h', mode='r')
            out_file_path = os.path.join(wav_dir, os.path.splitext(os.path.basename(pcm_file))[0] + ".wav")
            write(out_file_path, 16000, data)
            script_file = os.path.splitext(pcm_file)[0] + ".txt"
            f = open(script_file, encoding="euc-kr")
            script = f.readline()
            script = re.sub(r"[a-z]/[ ]?", "", script).strip()
            f.close()
            output_file.write(out_file_path + "|" + script + "\n")
output_file.close()