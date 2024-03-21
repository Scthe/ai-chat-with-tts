from src.inject_external_torch_into_path import inject_path

inject_path()

from typing import Optional
from TTS.api import TTS
from termcolor import colored
from tqdm import tqdm
import click
import os.path

from src.tts_utils import (
    exec_tts_to_file,
    list_speakers as list_speakers_util,
    get_torch_device,
)
from src.config import load_app_config

DEFAULT_TEXT = "The current algorithm only upscales the luma, the chroma is preserved as-is. This is a common trick known"


def create_tts(model_name: Optional[str] = None, use_gpu=True):
    print(colored("TTS model:", "blue"), model_name)
    tts = TTS(model_name=model_name, gpu=use_gpu, progress_bar=True)
    print(colored("TTS device:", "blue"), get_torch_device(tts))
    return tts


@click.command()
def list_models():
    cfg = load_app_config()
    tts = create_tts(cfg.tts.model_name)

    model_mgr = tts.list_models()
    print(colored("TTS models:", "blue"))
    for model in model_mgr.list_models():
        print("\t-", model)


@click.command()
@click.argument("model_name", type=click.STRING, required=True)
def list_speakers(model_name):
    tts = create_tts(model_name)

    speakers = list_speakers_util(tts)
    if len(speakers) == 0:
        print(colored("Model has no speakers", "blue"))
        exit(0)

    print(colored("TTS speakers for", "blue"), f"{model_name}:")
    for speaker in speakers:
        print("\t-", speaker)


@click.command()
@click.argument("model_name", type=click.STRING, required=True)
@click.argument("cloned_wav_file", type=click.Path(exists=True), required=True)
def create_speaker_samples(model_name, cloned_wav_file):
    """Generate samples for voice cloning if the TTS model has many speakers"""

    tts = create_tts(model_name)
    if not tts.is_multi_speaker:
        print(colored("Model is not multispeaker", "red"))
        exit(1)

    speakers = list_speakers_util(tts)
    print(colored("TTS speakers:", "blue"), speakers)

    text = DEFAULT_TEXT
    print(colored("Sample sentence:", "blue"), f"'{text}'")

    out_dir = "out_speaker_samples"
    print(colored("Will write to:", "blue"), f"'{out_dir}'")
    os.makedirs(out_dir, exist_ok=True)

    for speaker in tqdm(speakers):
        # print(speaker)
        out_file = os.path.join(out_dir, f"out_{speaker}.wav")

        if os.path.exists(out_file):
            # print(f"SKIP: {speaker}")
            continue
        tts.tts_with_vc_to_file(
            text=text,
            file_path=out_file,
            speaker=speaker,
            speaker_wav=cloned_wav_file,
            language="en",
        )


@click.command()
@click.option("--config", "-c", help="Config file")
@click.option("--input", "-t", help="Text to say")
@click.option(
    "--voice", "-v", type=click.Path(exists=True), help="Cloned voice, overrides config"
)
def speak(config: str, input: str, voice: Optional[str]):
    """Speak the text and write the result into the file. Can also voice-clone."""

    cfg = load_app_config(config)
    if voice != None:
        cfg.tts.sample_of_cloned_voice_wav = voice
    print(colored("Config", "blue"), cfg)

    text = input if input != None else DEFAULT_TEXT
    print(colored("Text to say:", "blue"), f"'{text}'")

    tts = create_tts(cfg.tts.model_name)

    out_file_path = "out_speak_result.wav"
    print(colored("Will write result to:", "blue"), out_file_path)

    exec_tts_to_file(cfg, tts, text, out_file_path, verbose=True)


@click.group()
def main():
    """Scripts/demos for TTS library."""


if __name__ == "__main__":
    main.add_command(list_models)
    main.add_command(list_speakers)
    main.add_command(create_speaker_samples)
    main.add_command(speak)
    main()
