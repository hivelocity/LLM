#! /usr/bin/env python3
import os
import argparse
import sys
from threading import Thread
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.streamers import TextIteratorStreamer


DEVICE = 'cuda'


@dataclass
class BasicConfig:
    max_length: int
    temperature: float
    top_p: float
    repetition_penalty: float


@dataclass
class Prompter:
    sys_prompt: str
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    config: GenerationConfig = field(init=False)

    def __post_init__(self, from_config: Optional[Union[GenerationConfig, BasicConfig]] = None):
        if not from_config:
            from_config = BasicConfig(max_length=1024, temperature=1.1, top_p=0.95, repetition_penalty=1.0)

        if isinstance(from_config, BasicConfig):
            self.config = GenerationConfig(
                max_length=from_config.max_length,
                temperature=from_config.temperature,
                top_p=from_config.top_p,
                repetition_penalty=from_config.repetition_penalty,
                do_sample=True,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.eos_token_id,
                # TODO: is this required?
                transformers_version="4.34.0.dev0",
            )
        else:
            self.config = from_config

    def prompt(self, prompt: str) -> str:
        inputs = self._get_prompt_inputs(prompt)
        outputs = self.model.generate(**inputs, generation_config=self.config)
        return self.tokenizer.batch_decode(outputs)[0]

    def prompt_iter(self, prompt: str) -> Iterator[str]:
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        inputs = self._get_prompt_inputs(prompt)

        generate_kwargs = dict(generation_config=self.config, streamer=streamer)
        generate_kwargs.update(inputs)

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

        thread.join(30)

    def _get_prompt_inputs(self, prompt: str):
        prefix = "<|im_start|>"
        suffix = "<|im_end|>\n"
        sys_format = prefix + "system\n" + self.sys_prompt + suffix
        user_format = prefix + "user\n" + prompt + suffix
        assistant_format = prefix + "assistant\n"
        input_text = sys_format + user_format + assistant_format

        return self.tokenizer(input_text, return_tensors="pt", return_attention_mask=True).to(DEVICE)


def create_model_and_tokenizer(model_path: str, *, half: bool) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    if half:
        model = model.half()
    model = model.to(DEVICE)

    return model, tokenizer


def get_default_model_path():
    models_dir = "/models"
    try:
        # List directories only
        dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        if len(dirs) == 1:
            return os.path.join(models_dir, dirs[0])
        else:
            raise ValueError("There should be exactly one directory in /models for a default value.")
    except Exception as e:
        raise e


def main() -> int:
    parser = argparse.ArgumentParser(description='Program to demonstrate argparse with a default directory.')
    parser.add_argument('-m', '--model', type=str, default=get_default_model_path(),
                        help='Path to the model directory. Defaults to the only folder in /models if not specified.')
    parser.add_argument('--half', default=False, action='store_true', help='loads model as half-precision')
    parser.add_argument('-p', '--prompt', type=str, default='A chat.')
    parser.add_argument('--max-length', '-l', type=int, default=1024)
    args = parser.parse_args()

    model_path = args.model
    half = args.half
    sys_prompt = args.prompt
    max_length = args.max_length
    print(args)

    config = BasicConfig(max_length=max_length, temperature=1.1, top_p=0.95, repetition_penalty=1.0)
    print(config)

    model, tokenizer = create_model_and_tokenizer(model_path, half=half)
    prompter = Prompter(sys_prompt=sys_prompt, model=model, tokenizer=tokenizer)

    while True:
        try:
            input_ = input('> ').strip()
        except KeyboardInterrupt:
            break

        if not input_ or input_.lower() in ('exit', 'quit', 'q'):
            break

        print()
        for text in prompter.prompt_iter(input_):
            sys.stdout.write(text)
            sys.stdout.flush()
        print()

if __name__ == '__main__':
    sys.exit(main())
