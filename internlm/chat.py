

from queue import Queue
from typing import Optional
from threading import Thread

from transformers.generation.streamers import BaseStreamer


def init_model_args(model_args=None):
    if model_args is None:
        model_args = {}
    model_args['temperature'] = model_args['temperature'] if model_args.get('temperature') != None else 0.95
    if model_args['temperature'] <= 0:
        model_args['temperature'] = 0.1
    if model_args['temperature'] > 1:
        model_args['temperature'] = 1
    model_args['top_p'] = model_args['top_p'] if model_args.get('top_p') else 0.7
    model_args['max_tokens'] = model_args['max_tokens'] if model_args.get('max_tokens') != None else 513

    return model_args


def do_chat_stream(model, tokenizer, question, history, model_args=None):
    model_args = init_model_args(model_args)
    streamer = ChatStreamer(tokenizer)
    kwargs = dict(
        tokenizer=tokenizer,
        query=question,
        streamer=streamer,
        history=history,
        temperature=model_args['temperature'],
        top_p=model_args['top_p'],
        max_new_tokens=max(2048, model_args['max_tokens'])
    )
    thread = Thread(target=model.chat, kwargs=kwargs)
    thread.start()

    for i, response in enumerate(streamer):
        if i == 0:
            # filter >:
            continue
        yield response


def do_chat(model, tokenizer, question, history, model_args=None):
    model_args = init_model_args(model_args)
    response, _ = model.chat(
        tokenizer=tokenizer,
        query=question,
        history=history,
        temperature=model_args['temperature'],
        top_p=model_args['top_p'],
        max_new_tokens=max(2048, model_args['max_tokens'])
    )
    return response


class ChatStreamer(BaseStreamer):
    def __init__(self, tokenizer, timeout: Optional[float] = None) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.text_queue = Queue()
        self.stop_signal = None
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("ChatStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]
        token = self.tokenizer.decode([value[-1]], skip_special_tokens=True)
        if token.strip() != "<eoa>":
            self.on_finalized_text(token)

    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        self.on_finalized_text('', stream_end=True)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Put the new text in the queue. If the stream is ending, also put a stop signal in the queue."""
        if stream_end:
            self.text_queue.put(self.stop_signal, timeout=self.timeout)
        else:
            self.text_queue.put(text, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.text_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value
