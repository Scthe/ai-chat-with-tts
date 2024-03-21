import traceback
from aiohttp import web
from TTS.api import TTS
from ollama import AsyncClient as OllamaAsyncClient
from termcolor import colored
from typing import Any

from src.config import AppConfig
from src.utils import Timer
from src.tts_utils import exec_tts, wav2bytes


def wrap_prompt_gemma2b(text, **kwargs):
    GEMMA_TEMPLATE = """<start_of_turn>user
    Answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    
    Question: {text}<end_of_turn>
    <start_of_turn>model"""

    return GEMMA_TEMPLATE.format(text=text, **kwargs)


class SocketMsgHandler:
    """
    https://github.com/Scthe/rag-chat-with-context/blob/master/src/socket_msg_handler.py
    """

    def __init__(
        self,
        cfg: AppConfig,
        llm: OllamaAsyncClient,
        tts: TTS,
        ws: web.WebSocketResponse,
    ):
        self.cfg = cfg
        self.ws = ws
        self.tts = tts
        self.llm = llm

    async def __call__(self, msg):
        msg_id = msg.get("msgId", "")
        type = msg.get("type", "")

        try:
            if type == "query":
                await self.ask_question(msg_id, msg)
            else:
                print(
                    colored(f'[Socket error] Unrecognised message: "{type}"', "red"),
                    msg,
                )

        except Exception as e:
            traceback.print_exception(e)
            data = {
                "type": "error",
                "msgId": msg_id,
                "error": str(e),
            }
            await self.ws_send_json(data)

    async def ask_question(self, msg_id, msg):
        q = msg.get("text", "")
        not_voiced_tokens = []
        # TODO find better place to split. On space is ok. ATM it can cut on dates e.g. ['1','998']
        tts_chunk_size = self.cfg.tts.chunk_size

        with Timer() as llm_timer:
            answer_gen_async = self.ask_question_to_llm(q)
            # answer_gen_async = async_wrap_iter(answer_gen)
            async for token in answer_gen_async:
                data = {"type": "token", "msgId": msg_id, "token": token}
                await self.ws_send_json(data)

                not_voiced_tokens.append(token)
                if tts_chunk_size > 0 and len(not_voiced_tokens) > tts_chunk_size:
                    flush_now_tokens = not_voiced_tokens[:tts_chunk_size]
                    not_voiced_tokens = not_voiced_tokens[tts_chunk_size:]
                    self.tts_and_send_to_client(msg_id, flush_now_tokens)

        self.tts_and_send_to_client(msg_id, not_voiced_tokens)

        data = {
            "type": "done",
            "msgId": msg_id,
            "elapsed_llm": llm_timer.delta,
        }
        # print(data)
        await self.ws_send_json(data)

    async def ask_question_to_llm(self, q: str):
        """If this fn returns nothing, just restart ollama"""
        cfg = self.cfg.llm
        if not cfg.enabled:
            yield q
            return

        prompt = wrap_prompt_gemma2b(q)

        # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion
        stream = await self.llm.generate(
            model=cfg.model,
            prompt=prompt,
            stream=True,
            # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
            options={
                "temperature": self.cfg.llm.temperature,
                "top_k": self.cfg.llm.top_k,
                "top_p": self.cfg.llm.top_p,
            },
        )
        async for chunk in stream:  # type: ignore
            # print(chunk)
            token: str = chunk.get("response")  # type: ignore
            yield token

    def tts_and_send_to_client(self, msg_id, tokens):
        import asyncio

        # TODO https://docs.coqui.ai/en/latest/models/xtts.html#streaming-manually
        text = "".join(tokens)
        text = text.replace("*", "")  # unprononcable
        if len(text) <= 0:
            return
        # print(colored("voicing:", "green"), tokens)

        async def tts_internal():

            with Timer() as tts_timer:
                wav = exec_tts(self.cfg, self.tts, text)
                bytes = wav2bytes(self.tts, wav)

            # Remember: websockets are on TCP. Always in correct order.
            # TODO what if last chunk is small (processed fast) and is send before the 2nd-to-last?
            await self.ws_send_bytes(bytes)
            data = {
                "type": "tts-elapsed",
                "msgId": msg_id,
                "elapsed_tts": tts_timer.delta,
            }
            await self.ws_send_json(data)

        loop = asyncio.get_running_loop()
        # asyncio.run(tts_internal())
        loop.create_task(tts_internal())

    async def ws_send_json(self, data: Any):
        await self.ws.send_json((data))

    async def ws_send_bytes(self, data: Any):
        await self.ws.send_bytes(data)
