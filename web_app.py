import sys
sys.path.append('waveglow/')
# import glow
from flask import Flask, render_template, redirect, url_for, request, send_from_directory
import numpy as np
import os
from pydub import AudioSegment
from hparams import create_hparams
from train import load_model
import torch
from denoiser import Denoiser
from text import text_to_sequence
from scipy.io.wavfile import write
import uuid
import time

SYNTH_DIR = 'static/synth_wav'
tacotron2_model = None
waveglow_model = None
denoiser = None

def init_model():
    print("init model!!!!")
    global tacotron2_model
    global waveglow_model
    global denoiser

    tacotron2_path = "outdir_finetune/checkpoint_62500"
    #    tacotron2_path = "outdir_korean/checkpoint_8800"
    #    tacotron2_path = "../models/tacotron2/outdir_korean/checkpoint_25000"
    #    tacotron2_path = "../tacotron2-pytorch/outdir/checkpoint_15000"
    #    tacotron2_path = "../models/tacotron2/outdir_korean/checkpoint_15000"
    #    tacotron2_path = "outdir_lj_korean/checkpoint_5000"
    #    tacotron2_path = "outdir_longtrain/checkpoint_439500"
    waveglow_path = "../waveglow-fix/checkpoints_finetune/waveglow_478000"
    #   waveglow_path = "../waveglow/checkpoints/waveglow_335000"
    # waveglow_path = "../waveglow-fix/checkpoints_longtrain/waveglow_484000"
    sampling_rate = 22050
    denoiser_strength = 0.0
    hparams = create_hparams()
    hparams.sampling_rate = sampling_rate
    hparams.training = False

    tacotron2_model = load_model(hparams)
    tacotron2_model.load_state_dict(torch.load(tacotron2_path)['state_dict'])
    _ = tacotron2_model.cuda().eval().half()

    # with open("waveglow/config.json") as f:
    #     data = f.read()
    # import json
    # config = json.loads(data)
    # waveglow_config = config["waveglow_config"]
    #
    # waveglow_model = glow.WaveGlow(**waveglow_config)
    #
    # checkpoint_dict = torch.load(waveglow_path, map_location='cpu')
    # model_for_loading = checkpoint_dict['model']
    # waveglow_model.load_state_dict(model_for_loading.state_dict())
    #
    # # waveglow_model.load_state_dict(torch.load(waveglow_path)['state_dict'])
    # waveglow_model = waveglow_model.remove_weightnorm(waveglow_model)
    # waveglow_model.cuda().eval().half()

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


@app.route("/simple_synth")
def simple_synth():
    text = request.args.get('input_text', default=None, type=str)
    sigma = request.args.get('sigma', default=0.8, type=float)
    sampling_rate = request.args.get('sampling_rate', default=22050, type=int)
    denoiser_strength = request.args.get('denoiser_strength', default=0.0, type=float)

    if not text:
        return render_template("simple_synth.html", input_text=None, synth_wav_path=None,
                           sigma=sigma, sampling_rate=sampling_rate, denoiser_strength=denoiser_strength, elapsed=None)


    start = time.time()
    # sequence = np.array(text_to_sequence(text, ['transliteration_cleaners']))[None, :]
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    #    sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
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
    elapsed = time.time() - start
    filename = str(uuid.uuid4()) + '.wav'
    audio_path = os.path.join(SYNTH_DIR, filename)
    write(audio_path, sampling_rate, audio)

    return render_template("simple_synth.html", input_text=text,
                           synth_wav_path=filename,
                           sigma=sigma, sampling_rate=sampling_rate, denoiser_strength=denoiser_strength, elapsed=elapsed)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(directory=SYNTH_DIR, filename=filename, as_attachment=True)

@app.route('/download_mp3/<path:filename>', methods=['GET', 'POST'])
def download_mp3(filename):
    new_filename = os.path.splitext(filename)[0] + ".mp3"
    AudioSegment.from_wav(os.path.join(SYNTH_DIR, filename)).export(os.path.join(SYNTH_DIR, new_filename), format="mp3")
    return send_from_directory(directory=SYNTH_DIR, filename=new_filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, threaded=False)
