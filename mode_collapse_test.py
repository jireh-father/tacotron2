import os
import sys

sys.path.append('waveglow/')
from scipy.io.wavfile import write
import numpy as np
import torch
import zipfile
from hparams import create_hparams
from train import load_model
from text import text_to_sequence
from denoiser import Denoiser
import glob
import datetime
import random


def main(tacotron2_path, waveglow_path, sigma, output_dir, sampling_rate, denoiser_strength, text, file_idx,
         inference_name, zip_file):
    hparams = create_hparams()
    hparams.sampling_rate = sampling_rate

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)
    random.seed(hparams.seed)

    model = load_model(hparams)
    model.load_state_dict(torch.load(tacotron2_path)['state_dict'])
    _ = model.cuda().eval().half()

    waveglow = torch.load(waveglow_path)['model']
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow)

    sequence = np.array(text_to_sequence(text, ['transliteration_cleaners']))[None, :]
    print(sequence)
    # sequence2 = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
    # sequence3 = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
    # print(np.array_equal(sequence, sequence2))
    # print(np.array_equal(sequence, sequence3))
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    print(mel_outputs_postnet.shape)


    MAX_WAV_VALUE = 32768.0
    # print(mel_outputs_postnet.cpu().data.numpy()[0][0][:30])
    # print(mel_outputs_postnet2.cpu().data.numpy()[0][0][:30])
    # if np.array_equal(mel_outputs_postnet.cpu().data.numpy(), mel_outputs_postnet2.cpu().data.numpy()):
    #     print("same!!")
    # else:
    #     print("different!!")
    #
    # if np.array_equal(mel_outputs_postnet.cpu().data.numpy(), mel_outputs_postnet3.cpu().data.numpy()):
    #     print("same!!")
    # else:
    #     print("different!!")
    #
    with torch.no_grad():
        for i in range(10):
            audio = waveglow.infer(mel_outputs_postnet, sigma=sigma)  # 0.666)
            audio = audio.squeeze()
            audio = audio.cpu().numpy()
            audio = audio.astype('int16')
            print(audio.shape)

        # audio = audio.cpu().numpy()
        # audio2 = waveglow.infer(mel_outputs_postnet, sigma=sigma)  # 0.666)
        # audio2 = audio2.squeeze()
        # audio2 = audio2.cpu().numpy()
        #
        # audio3 = waveglow.infer(mel_outputs_postnet, sigma=sigma)  # 0.666)
        # audio3 = audio3.squeeze()
        # audio3 = audio3.cpu().numpy()
        #
        # print(audio[:30])
        # print(audio2[:30])
        # print(audio3[:30])
        #
        # if np.array_equal(audio, audio2):
        #     print("same!!")
        # else:
        #     print("different!!")
        #
        # if np.array_equal(audio, audio3):
        #     print("same!!")
        # else:
        #     print("different!!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tacotron2_path')
    parser.add_argument('-w', '--waveglow_path',
                        help='Path to waveglow decoder checkpoint with model')
    parser.add_argument('-o', "--output_dir", required=True)
    parser.add_argument('-f', "--textlist_path", required=True)
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument("--sampling_rate", default=22050, type=int)
    parser.add_argument("-d", "--denoiser_strength", default=0.0, type=float,
                        help='Removes model bias. Start with 0.1 and adjust')

    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.textlist_path, encoding="utf-8") as f:
        text_list = f.readlines()

    if os.path.isdir(args.tacotron2_path):
        t2_cp_paths = glob.glob(os.path.join(args.tacotron2_path, "checkpoint_*"))
    else:
        t2_cp_paths = args.tacotron2_path.split(",")

    if os.path.isdir(args.waveglow_path):
        wg_cp_paths = glob.glob(os.path.join(args.waveglow_path, "waveglow_*"))
    else:
        wg_cp_paths = args.waveglow_path.split(",")


    zip_name = "t2_%d_wg_%d_%s" %(len(t2_cp_paths), len(wg_cp_paths), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    zip_file = zipfile.ZipFile(
        os.path.join(args.output_dir, zip_name + ".zip"), 'w')
    for t2_model_path in t2_cp_paths:
        for wg_model_path in wg_cp_paths:
            t2_steps = os.path.basename(t2_model_path).split("_")[1]
            wg_steps = os.path.basename(wg_model_path).split("_")[1]
            infer_name = "t2_%s_wg_%s" % (t2_steps, wg_steps)
            # zip_file = zipfile.ZipFile(
            #     os.path.join(args.output_dir, infer_name + ".zip"), 'w')
            for i, text in enumerate(text_list):
                main(t2_model_path, wg_model_path, args.sigma, args.output_dir,
                     args.sampling_rate, args.denoiser_strength, text, i, infer_name, zip_file)

            # zip_file.close()
    zip_file.close()
