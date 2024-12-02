import os
import torch
import pytest
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

nicknames = {
        "7B": "mistralai/Mistral-7B-v0.1",
        "7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.2",
        "8x7B": "mistralai/Mixtral-8x7B-v0.1",
        "8x7B-Instruct": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "8x22B": "mistralai/Mixtral-8x22B-v0.1",
        "8x22B-Instruct": "mistralai/Mixtral-8x22B-Instruct-v0.1"
    }

#### TOKENIZER ####

def load_tokenizer(model_nickname):
    model_id = nicknames[model_nickname] if model_nickname in nicknames else model_nickname
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding="max_length",
        truncation=False,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@pytest.mark.parametrize(
    "model_nickname",
    [
        "7B",
        "7B-Instruct",
        "8x7B",
        "8x7B-Instruct",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mixtral-8x7B-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
    ]
)
def test_load_tokenizer(model_nickname):
    model_id = nicknames[model_nickname] if model_nickname in nicknames else model_nickname
    tokenizer = load_tokenizer(model_id)
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "right"

#### MODEL ####

def load_mistral(model_nickname):
    model_id = nicknames[model_nickname] if model_nickname in nicknames else model_nickname
    print(f"Loading model {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        # torch_dtype=torch.float16,
        load_in_8bit=True if "8x7B" in model_id else False,
        load_in_4bit=True if "8x22B" in model_id else False,
        token = os.getenv("HF_TOKEN")
    )
    return model