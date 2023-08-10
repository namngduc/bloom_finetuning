#!/usr/bin/env python3

import torch
import os
import math
from transformers import GenerationConfig
from fastapi import FastAPI
import logging
import uvicorn
from typing import List
from pydantic import BaseModel
from load_model import model, tokenizer, device


try:
    batch = os.environ["BATCH_SIZE"]
except KeyError:
    batch = 2                          # Default batch_size = 2


class Item(BaseModel):
    instruction: str
    input: str | None = None


class ModelInference:
    def __init__(self, model) -> None:
        self.model = model
        self.batch_size = int(batch)

    
    def generate_prompt(self, instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 
            ### Instruction:
            {instruction}
            ### Input:
            {input}
            ### Response:
            """
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            {instruction}
            ### Response:
            """

    def make_infer(self, input_sentences, 
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            **kwargs,
        ):
        if self.batch_size > len(input_sentences):
            # Dynamically extend to support larger bs by repetition
            input_sentences *= math.ceil(self.batch_size / len(input_sentences))
        items = input_sentences[: self.batch_size]
        prompts = []
        for item in items:
            instruct = item["instruction"]
            input = item["input"]
            prompt = self.generate_prompt(instruct, input)
            prompts.append(prompt)
        input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt",  padding=True)
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        generate_kwargs = dict(generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens, 
                do_sample=False)
        with torch.no_grad():
            generation_output = self.model.generate(
                **input_tokens,
                **generate_kwargs
            )
        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        instructions, inputs, responses = [], [], []
        for output in outputs:
            instruction = output.split("### Instruction")[1].split("\n")[1].strip()
            inp = output.split("### Input")
            input = inp[1].split("\n")[1].strip() if len(inp) > 1 else ""
            response = output.split("### Response:")[1].strip()
            instructions.append(instruction)
            inputs.append(input)
            responses.append(response)
        return zip(instructions, inputs, responses)


app = FastAPI()
inference = ModelInference(model)


@app.on_event("startup")
async def startup_event():
    logger_access = logging.getLogger("uvicorn.access")
    logger_error = logging.getLogger("uvicorn.error")
    console_formatter = uvicorn.logging.ColourizedFormatter(
        "{asctime} [{name}] {levelprefix} {message}",
        style="{", use_colors=True)
    logger_access.handlers[0].setFormatter(console_formatter)
    logger_error.handlers[0].setFormatter(console_formatter)


@app.post("/bloom_generate/")
async def create_item(items: List[Item]):
    input_sentences = []
    for item in items:
        item_dict = item.dict()
        input_sentences.append(item_dict)
    responses = inference.make_infer(input_sentences)
    results = []
    for (instruction, input, output) in responses:
        result = {
            'instuction': instruction,
            'input': input,
            'output': output 
        }
        results.append(result)
    return results 
