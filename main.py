from src.inject_external_torch_into_path import inject_path

inject_path()

from termcolor import colored
import argparse
from TTS.api import TTS
from ollama import AsyncClient

from src.server import create_server, set_socket_msg_handler, start_server
from src.socket_msg_handler import SocketMsgHandler
from src.config import load_app_config
from src.tts_utils import get_torch_device

STATIC_DIR = "./static"
DEFAULT_CONFIG_FILE = "config.yaml"

if __name__ == "__main__":
    # https://github.com/Scthe/rag-chat-with-context/blob/master/main.py#L175

    parser = argparse.ArgumentParser(
        description=f"LLM chat with voice-over",
    )
    parser.add_argument("--config", "-c", help="Config file")
    args = parser.parse_args()

    cfg_file = args.config if args.config else DEFAULT_CONFIG_FILE
    cfg = load_app_config(cfg_file)
    print(colored("Config:", "blue"), cfg)

    llm = AsyncClient(cfg.llm.api)

    app = create_server(STATIC_DIR)

    # TODO use `cfg.tts.vocoder_name`
    tts = TTS(model_name=cfg.tts.model_name, gpu=cfg.tts.use_gpu)
    print(colored("TTS device:", "blue"), get_torch_device(tts))

    handler = lambda ws: SocketMsgHandler(cfg, llm, tts, ws)
    set_socket_msg_handler(app, handler)

    start_server(app, host=cfg.server.host, port=cfg.server.port)

    print("=== DONE ===")
