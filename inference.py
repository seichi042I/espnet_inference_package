import os
import time
import torch
import pyopenjtalk
from espnet2.bin.tts_inference import Text2Speech
from espnet2.text.phoneme_tokenizer import pyopenjtalk_g2p
import matplotlib.pyplot as plt
from espnet2.text.token_id_converter import TokenIDConverter
import numpy as np

import argparse
import yaml
import torchaudio


prosodic = False

parser = argparse.ArgumentParser()

parser.add_argument("--model_tag")
parser.add_argument("--train_config")
parser.add_argument("--model_file")
parser.add_argument("--vocoder_tag")
parser.add_argument("--vocoder_config")
parser.add_argument("--vocoder_file")
parser.add_argument("-p", "--prosodic",
                    help="Prosodic text input mode", action="store_true")
parser.add_argument(
    "--preg2p", help="preprocessing g2p use true or false", action="store_true")
parser.add_argument("--fs", type=int, default=24000)
parser.add_argument("--reference", action="store_true")

args = parser.parse_args()

# Case 2: Load the local model and the pretrained vocoder
print("download model = ", args.model_tag, "\n")
print("download vocoder = ", args.vocoder_tag, "\n")
print("モデルを読み込んでいます...\n")
if args.model_tag is not None :
    text2speech = Text2Speech.from_pretrained(
        model_tag=args.model_tag,
        vocoder_tag=args.vocoder_tag,
        device="cuda:1",
    )
elif args.vocoder_tag is not None :
    text2speech = Text2Speech.from_pretrained(
        train_config=args.train_config,
        model_file=args.model_file,
        vocoder_tag=args.vocoder_tag,
        device="cuda:1",
    )
else :
    text2speech = Text2Speech.from_pretrained(
        train_config=args.train_config,
        model_file=args.model_file,
        vocoder_config=args.vocoder_config,
        vocoder_file=args.vocoder_file,
        device="cuda:1",
    )
with open(args.train_config) as f:
    config = yaml.safe_load(f)
config = argparse.Namespace(**config)

with open(config.train_data_path_and_name_and_type[1][0]) as f:
    lines = f.readlines()

guide = "セリフを入力してください"
if args.prosodic :
    guide = "アクセント句がスペースで区切られた韻律記号(^)付きのセリフをすべてひらがなで入力してください。(スペースや記号もすべて全角で)\n"
x = ""
while (1):
    # decide the input sentence by yourself
    print(guide)

    x = input()
    if x == "exit" :
        break
    if args.reference:
        print("wav.scpの行番号を入力してください")
        refer_id = input()
        path_str = lines[int(refer_id)].split(' ')[1].strip()
        speech, speech_fs = torchaudio.load(path_str)
        print(path_str)

    # model, train_args = TTSTask.build_model_from_file(
    #        args.train_config, args.model_file, "cuda"
    #        )

    token_id_converter = TokenIDConverter(
        token_list=text2speech.train_args.token_list,
        unk_symbol="<unk>",
    )

    if args.preg2p:
        token = pyopenjtalk_g2p(x)
        text_ints = token_id_converter.tokens2ids(token)
        text = np.array(text_ints)
    else:
        text = x

    # synthesis
    with torch.no_grad():
        start = time.time()
        if config.tts_conf["generator_params"]["use_gst"]:
            feats, _ = text2speech.feats_extract(speech.to(torch.device("cuda:1")))
            data = text2speech(text, feats=feats)
        else:
            data = text2speech(text)
        wav = data["wav"]
        # print(text2speech.preprocess_fn("<dummy>",dict(text=x))["text"])
    rtf = (time.time() - start) / (len(wav) / text2speech.fs)
    print(f"RTF = {rtf:5f}")

    if not os.path.isdir("generated_wav"):
        os.makedirs("generated_wav")

    if args.model_tag is not None :
        if "tacotron" in args.model_tag :
            mel = data['feat_gen_denorm'].cpu()
            plt.imshow(torch.t(mel).numpy(),
                       aspect='auto',
                       origin='lower',
                       interpolation='none',
                       cmap='viridis'
                       )
            plt.savefig('generated_wav/' + x + '.png')
    else :
        if "tacotron" in args.model_file :
            mel = data['feat_gen_denorm'].cpu()
            plt.imshow(torch.t(mel).numpy(),
                       aspect='auto',
                       origin='lower',
                       interpolation='none',
                       cmap='viridis'
                       )
            plt.savefig('generated_wav/' + x + '.png')
        if "fastspeech2" in args.model_file:
            print(data["pitch"].squeeze())

    # let us listen to generated samples
    from IPython.display import display, Audio
    import numpy as np
    #display(Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs))
    #Audio(wav.view(-1).cpu().numpy(), rate=text2speech.fs)
    np_wav = wav.view(-1).cpu().numpy()

    print("サンプリングレート", args.fs, "で出力します。")
    from scipy.io.wavfile import write
    samplerate = args.fs
    t = np.linspace(0., 1., samplerate)
    amplitude = np.iinfo(np.int16).max
    data = amplitude * np_wav / np.max(np.abs(np_wav))
    write("generated_wav/" + x + ".wav", samplerate, data.astype(np.int16))
    print("\n\n\n")
