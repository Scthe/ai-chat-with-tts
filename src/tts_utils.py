from TTS.api import TTS
from src.config import AppConfig
from termcolor import colored


def get_torch_device(tts: TTS):
    model = tts.synthesizer.tts_model
    return next(model.parameters()).device


def list_speakers(tts: TTS):
    sm = tts.synthesizer.tts_model.speaker_manager
    if sm == None:
        return []
    return list(tts.synthesizer.tts_model.speaker_manager.name_to_id)


def get_tts_options(cfg: AppConfig, tts: TTS):
    kw = {}
    is_cloning = False

    if tts.is_multi_speaker:
        kw["speaker"] = cfg.tts.speaker
    if tts.is_multi_lingual:
        kw["language"] = cfg.tts.language
    # emotion: str = None,
    # speed: float = None,
    # split_sentences: bool = True,

    sample_of_cloned_voice_wav = cfg.tts.sample_of_cloned_voice_wav
    if sample_of_cloned_voice_wav:
        is_cloning = True
        kw["speaker_wav"] = sample_of_cloned_voice_wav

    return is_cloning, kw


def exec_tts(cfg: AppConfig, tts: TTS, text: str):
    is_cloning, tts_kwargs = get_tts_options(cfg, tts)

    if is_cloning:
        wav = tts.tts_with_vc(text=text, **tts_kwargs)
    else:
        wav = tts.tts(text=text, **tts_kwargs)

    return wav


def exec_tts_to_file(
    cfg: AppConfig, tts: TTS, text: str, out_file_path: str, verbose=False
):
    is_cloning, tts_kwargs = get_tts_options(cfg, tts)

    if is_cloning:
        if verbose:
            print(colored("Cloning voice based on:", "blue"), f"'{tts_kwargs["speaker_wav"]}'")
        wav = tts.tts_with_vc_to_file(text=text, file_path=out_file_path, **tts_kwargs)
    else:
        if verbose:
            print(colored("Voice cloning:", "blue"), "OFF")
        wav = tts.tts_to_file(text=text, file_path=out_file_path, **tts_kwargs)

    return wav


def wav2bytes(tts: TTS, wav):
    import io

    out = io.BytesIO()
    tts.synthesizer.save_wav(wav, out)
    return out.getbuffer()
