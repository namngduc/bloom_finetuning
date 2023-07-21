#!/usr/bin/env python3

import torch
from transformers import GenerationConfig
import gradio as gr
from load_model import model, tokenizer, device
    
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. # noqa: E501
        ### Instruction:
        {instruction}
        ### Input:
        {input}
        ### Response:
        """
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. # noqa: E501
        ### Instruction:
        {instruction}
        ### Response:
        """

def make_infer(instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return output.split("### Response:")[1].strip()


if __name__ == "__main__":
    demo = gr.Interface(
        fn=make_infer,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder="Tell me about alpacas."
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            gr.components.Slider(minimum=0, maximum=100, step=1, value=40, label="Top k"),
            gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="🌲 🌲 🌲 BLOOM-Zalo",
        description="BLOOM-Zalo is a 1B1-parameter BLOOM model finetuned to follow instructions. It is trained on the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset and custom zalo dataset, makes use of the Huggingface LLaMA implementation. For more information, please visit [my project's website](https://github.com/namngduc/bloom_finetuning).",
    )
    demo.launch(server_name="0.0.0.0")