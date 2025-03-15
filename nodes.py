import os
import torch
import random
import numpy as np
import folder_paths
from transformers import AutoModelForCausalLM, AutoTokenizer
from comfy.model_management import get_torch_device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QwQModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["Qwen/QwQ-32B", ], {"default": "Qwen/QwQ-32B"}),
                "load_local_model": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "local_qwq_model_path": ("STRING", {"default": "Qwen/QwQ-32B"}),
            }
        }

    RETURN_TYPES = ("MODEL", "TOKENIZER")
    RETURN_NAMES = ("model", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "QwQ"

    def load_model(self, model_name, load_local_model, *args, **kwargs):
        device = get_torch_device()

        if load_local_model:
            # 如果加载本地模型，直接使用用户提供的路径
            model_name = kwargs.get("local_qwq_model_path", model_name)
            cache_dir = None  # 本地模型不需要指定 cache_dir
        else:
            # 如果加载 Hugging Face 模型，下载到 ComfyUI 的模型目录
            qwq_dir = os.path.join(folder_paths.models_dir, "QWQ32B")
            os.makedirs(qwq_dir, exist_ok=True)
            cache_dir = qwq_dir

        # 加载模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype="auto", device_map="auto"
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        return (model, tokenizer)


class QwQTextGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "tokenizer": ("TOKENIZER",),
                "prompt": ("STRING", {"default": "How many r's are in the word \"strawberry\""}),
                "max_new_tokens": ("INT", {"default": 32768, "min": 1, "max": 32768}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0, "max": 2}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 101}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate_text"
    CATEGORY = "QwQ"

    def generate_text(self, model, tokenizer, prompt, max_new_tokens, seed=0, temperature=1.0, max_tokens=500, top_k=50,
                      top_p=1.0, ):
        set_seed(seed % 9999999)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        """
        Use Temperature=0.6, TopP=0.95, MinP=0 instead of Greedy decoding to avoid endless repetitions.
        Use TopK between 20 and 40 to filter out rare token occurrences while maintaining the diversity of the generated output.
        For supported frameworks, you can adjust the presence_penalty parameter between 0 and 2 to reduce endless repetitions. However, using a higher value may result in occasional language mixing and a slight decrease in performance."""
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return (response,)


NODE_CLASS_MAPPINGS = {
    "QwQModelLoader": QwQModelLoader,
    "QwQTextGenerator": QwQTextGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QweQModelLoader": "QwQ Model Loader",
    "QwQTextGenerator": "QwQ Text Generator"
}
