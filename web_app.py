import sys
sys.path.append('waveglow/')
from flask import Flask, render_template, redirect, url_for, request
import numpy as np
import os

from hparams import create_hparams
from train import load_model
import torch
from denoiser import Denoiser
from text import text_to_sequence
from scipy.io.wavfile import write
import uuid

SYNTH_DIR = 'static/synth_wav'

def init_model():
    print("init model!!!!")
    global tacotron2_model
    global waveglow_model
    global denoiser

    tacotron2_path = "outdir_finetune/checkpoint_62500"
    waveglow_path = "../waveglow-fix/checkpoints_finetune/waveglow_478000"
    sampling_rate = 22050
    denoiser_strength = 0.0
    hparams = create_hparams()
    hparams.sampling_rate = sampling_rate

    tacotron2_model = load_model(hparams)
    tacotron2_model.load_state_dict(torch.load(tacotron2_path)['state_dict'])
    _ = tacotron2_model.cuda().eval().half()

    waveglow_model = torch.load(waveglow_path)['model']
    waveglow_model = waveglow_model.remove_weightnorm(waveglow_model)
    waveglow_model.cuda().eval().half()
    for k in waveglow_model.convinv:
        k.float()
    if denoiser_strength > 0:
        denoiser = Denoiser(waveglow_model)

def create_app():
    app = Flask(__name__)
    def run_on_start():
        init_model()
    run_on_start()
    return app
app = create_app()
# app = Flask(__name__)

tacotron2_model = None
waveglow_model = None
denoiser = None




@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return "Hello, %s!" % name


@app.route("/simple_synth")
def simple_synth():
    text = request.args['input_text']
    sigma = float(request.args['sigma'])
    sampling_rate = int(request.args['sampling_rate'])
    denoiser_strength = float(request.args['denoiser_strength'])

    if not text:
        return render_template("simple_synth.html", input_text=None, synth_wav_path=None,
                           sigma=sigma, sampling_rate=sampling_rate, denoiser_strength=denoiser_strength)

    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = tacotron2_model.inference(sequence)
    MAX_WAV_VALUE = 32768.0

    with torch.no_grad():
        audio = waveglow_model.infer(mel_outputs_postnet, sigma=sigma)  # 0.666)
        if denoiser_strength > 0:
            audio = denoiser(audio, denoiser_strength)  # 0.01 > denoiser_strength
        audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    audio = audio.astype('int16')
    if not os.path.isdir(SYNTH_DIR):
        os.makedirs(SYNTH_DIR)
    filename = str(uuid.uuid4()) + '.wav'
    audio_path = os.path.join(SYNTH_DIR, filename)
    write(audio_path, sampling_rate, audio)

    return render_template("simple_synth.html", input_text=text,
                           synth_wav_path=os.path.join(os.path.basename(SYNTH_DIR), filename),
                           sigma=sigma, sampling_rate=sampling_rate, denoiser_strength=denoiser_strength)


#
# @app.route('/uploads/<path:filename>', methods=['GET', 'POST'])
# def download(filename):
#     uploads = os.path.join(current_app.root_path, app.config['UPLOAD_FOLDER'])
#     return send_from_directory(directory=uploads, filename=filename)


if __name__ == "__main__":
    app.run()
