import os
import argparse


if __name__ == "__main__":
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', "--wav_dir", required=True)
    parser.add_argument('-s', '--script_path', required=True)
    parser.add_argument('-o', '--out_script_path', required=True)
    args = parser.parse_args()
    # wav_dir = "/home/ilseo/dataset/kss/wav"
    # script_path = "/home/ilseo/dataset/kss/transcript.v.1.2.txt"
    # out_script_path = "/home/ilseo/dataset/kss/script_for_taco2_pyroch.txt"
    wav_dir = args.wav_dir
    script_path = args.script_path
    out_script_path = args.out_script_path

    out_file = open(out_script_path, "w+", encoding="utf-8")

    with open(script_path) as f:
        for line in f:
            split_line = line.split("|")
            file_path = split_line[0] + ".pt"
            script = split_line[1]
            out_file.write(os.path.join(wav_dir, file_path) + "|" + script + "\n")

    out_file.close()