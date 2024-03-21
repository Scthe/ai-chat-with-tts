pip-freeze:
	pip freeze > requirements.txt

pip-install:
	pip install -r requirements.txt

start:
	python.exe main.py

start-xtts:
	python.exe main.py --config "config_xtts.yaml"

# ------------- UTILS:
list-models:
	python.exe tts_scripts.py list-models

xtts-list-speakers:
	python.exe tts_scripts.py list-speakers "tts_models/multilingual/multi-dataset/xtts_v2"
	
xtts-create-speaker-samples:
	python.exe tts_scripts.py create-speaker-samples "tts_models/multilingual/multi-dataset/xtts_v2" "voice_to_clone.wav"

xtts-speak-test:
	python.exe tts_scripts.py speak -c "config_xtts.yaml"

xtts-clone-test:
	python.exe tts_scripts.py speak -c "config_xtts.yaml" -v "voice_to_clone.wav"
