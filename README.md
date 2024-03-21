# AI chat (LLM) with text to speech (TTS)

Testing [xtts_v2](https://github.com/coqui-ai/TTS) when chatting with a Large language model (LLM). Ask the AI a question and the answer is read back to you.

> TODO video with audio
> **Video description**

## Usage

### Install ollama to access LLM models

1. Download ollama from [https://ollama.com/download](https://ollama.com/download).
2. `ollama pull gemma:2b`. Pull model file e.g. [gemma:2b](https://ollama.com/library/gemma:2b).
3. Verification:
   1. `ollama show gemma:2b --modelfile`. Inspect model file data.
   2. `ollama run gemma:2b`. Open the chat in the console to check everything is OK.

### Running this app

1. `pip install -r requirements.txt`. Install dependencies.
2. `pip install TTS`. There is a chance you already have TTS installed, as it's required to train your models. In that case, check below for `inject_external_torch_into_path.py`. Otherwise, this step is required. Python 3.10 is recommended, TTS has some checks for this (fails on 3.12.2).
3. (Optional) [Install CUDA-enabled PyTorch](https://pytorch.org/get-started/locally/).
4. `python.exe main.py --config "config_xtts.yaml"`. Start the app using [./config_xtts.yaml](config_xtts.yaml).
5. Go to [http://localhost:8080/index.html](http://localhost:8080/index.html).

Alternatively, use `python.exe main.py --config "config.yaml"` for a much smaller `tacotron2-DDC` TTS model.

If you don't want to install PyTorch with CUDA again (2.7+ GB), add the correct directory to [./src/inject_external_torch_into_path.py](src/inject_external_torch_into_path.py)

### Config

You can find the config value descriptions in [./config.yaml](config.yaml). The most important are:

- `llm.enabled: False`. Disable LLM. The TTS will read back the question.
- `llm.model: 'gemma:2b'`. Change LLM to something from Ollama's [library](https://ollama.com/library).
- `tts.use_gpu: False`. Use CPU for text to speech.
- `tts.model_name: 'tts_models/multilingual/multi-dataset/xtts_v2'`. Change the TTS model. Run `python.exe tts_scripts.py list-models` for a full list of available models.

### Enabling voice cloning

In [./config_xtts.yaml](config_xtts.yaml) set `tts.sample_of_cloned_voice_wav` to point to WAV audio. E.g. `sample_of_cloned_voice_wav: 'voice_to_clone.wav'`. Requirements:

- 16-bit, mono PCM WAV.
- Sample rate: 22050 Hz.
- Clean audio. No noise, distortion, etc.
- No silence at the start or end.
- At least 6s, preferably a bit longer.

[Audacity](https://www.audacityteam.org/) for audio editing works fine.

### Other commands

[./main.py](main.py) starts the server. Various util tools are in [./tts_scripts.py](tts_scripts.py) (see examples in [./makefile](makefile)):

- `python.exe tts_scripts.py list-models`. List TTS models.
- `python.exe tts_scripts.py list-speakers <model_name>`. List speakers for a selected model.
- `python.exe tts_scripts.py create-speaker-samples <model_name> <voice_to_clone.wav>`. Create audio samples for the cloned voice for all available speakers in the model. Use it to select the best speaker. This may take some time, see the code to adjust the spoken sentence.
  - Internally, the TTS package first renders the voice using the model's speaker. Then it uses [FreeVC](https://github.com/OlaWod/FreeVC) to clone it based on the provided `<voice_to_clone.wav>`.
- `python.exe tts_scripts.py speak -c "config_xtts.yaml"`. Speak a test sentence and write the result to a `.wav` file. No voice cloning. You can provide your config if you like.
- `python.exe tts_scripts.py speak -c "config_xtts.yaml" -v <voice_to_clone.wav>`. Speak a test sentence and write the result to a `.wav` file. Uses voice cloning based on provided `<voice_to_clone.wav>`. You can provide your config if you like.

## FAQ

**Q: Which models did you use?**

- Large language model (LLM):
  - `gemma-2b` (3B parameters). It returns proper sentences, so good enough.
- Text to speech (TTS):
  - `tacotron2-DDC`.
  - `xtts_v2`.

**Q: Why xtts_v2?**

It performed best for me when I did the Hugging Face's blind tests.

**Q: How is this app parallelized?**

There are 2 variants, based on `tts.chunk_size`:

- `tts.chunk_size: 0` (default). First, get a full answer from LLM. Then process it using TTS (which splits it into sentences due to potential memory limits). This introduces a delay before the text is spoken. The user will see the full answer before hearing the first word.
- `tts.chunk_size: 50` (or any other positive value). Generate 50 tokens from LLM, and send them to TTS (and later to the client). We are still generating tokens from LLM at the same time. Unfortunately, both operations at the same time are impossible on a single GPU. This option is nicer if you have 2 GPUs, or TTS is done on the CPU.

An interesting option is to use the GPU for LLM and the CPU for TTS. Unfortunately, depending on the TTS model, the CPU might struggle. And I'm not fluent in the Python threading model, but even if you push TTS to a separate thread, it will affect/starve the event loop. Yes, the code is already full async/await. No, it does not matter.

**Q: How fast is it?**

The first response is always the slowest. The models for LLM and TTS (and optional voice converter) have to be loaded into memory. LLM for me is near-instantious (RTX 3060 with 12GB VRAM and `gemma:2b`). TTS depends on the size of the model and voice cloning. `xtts_v2` with voice cloning (as seen in the video above) is the most expensive option.

**Q: I get nothing in response?**

1. Check that there are no other apps that have loaded models on GPU (video games, stable diffusion, etc.). Even if they don't do anything ATM, they still take VRAM.
2. Close Ollama.
3. Make sure VRAM usage is at 0.
4. Start Ollama.
5. Restart the app.
6. Ask a question to load all models into VRAM.
7. Check you are not running out of VRAM.

**Q: How can I list the speakers available in my model?**

See above the "Other commands" section.

## References

I've copied the whole UI and a lot of backend from my previous project: [retrieval-augmented generation with context](https://github.com/Scthe/rag-chat-with-context).

External packages:

- [TTS](https://docs.coqui.ai/en/latest/index.html). `xtts_v2` is the main reason for this project.
  - [Basic tutorial](https://docs.coqui.ai/en/latest/tutorial_for_nervous_beginners.html).
  - [Cheat sheet](https://docs.coqui.ai/en/latest/inference.html).
- [Ollama](https://ollama.com/). Used to run LLM server.
- [AIOHTTP](https://docs.aiohttp.org/en/stable/). Async Python server.
- [Preact](https://preactjs.com/). Cause I am lazy and can import it from CDN.
