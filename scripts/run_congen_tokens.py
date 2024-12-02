"""
Acceptability of GEN, ALL, MOST, SOME for sentences in CONGEN, for several token-based context windows.
"""
# CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m run_congen_tokens
# CUDA_VISIBLE_DEVICES=3,4,5 python -m run_congen_tokens

import os
import pandas as pd
from tqdm import tqdm
from utils.lm import load_tokenizer, load_mistral
from utils.congen import load_congen, find_document_by_id, get_left_contexts
from utils.acceptability import get_quantified_sentences, get_property_losses_from_context_quantified_sentences


model_nicknames = ["7B", "7B-Instruct"]
# model_nicknames = ["8x7B",  "8x7B-Instruct"]

sentences, documents = load_congen(congen_folder_path = "../data/congen")


for model_nickname in model_nicknames:
    print(f"Running model {model_nickname}")
    if model_nickname == "gemma7B":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto")
    else:
        model = load_mistral(model_nickname)
        tokenizer = load_tokenizer(model_nickname)
    debug = False
    full_sentence_loss = False
    output_file = f"results/congen_acceptability_{model_nickname}.csv"

    

    results = []
    for i, row in tqdm(sentences.iterrows(), total=len(sentences), desc=f"Running model {model_nickname}"):
        doc_row = find_document_by_id(documents, row["doc_id"])
        doc = doc_row["text"]

        # Build contexts
        token_splits = [0, 4, 8, 16, 24, 32]
        contexts = get_left_contexts(
            doc, 
            row["text"], 
            token_splits, 
            "tokens", 
            tokenizer
        )
        if not contexts:
            continue

        # Quantified sentences
        quantified_sentences = get_quantified_sentences(row['text'], row['original_quantifier'])

        for n_tokens_context in contexts.keys():
            context = contexts[n_tokens_context]
            plosses = get_property_losses_from_context_quantified_sentences(
                quantified_sentences, 
                model, 
                tokenizer, 
                context,
                full_sentence_loss = full_sentence_loss,
                debug = debug
            )

            results.append(
                {
                    "text": row['text'],
                    "original_quantifier": row['original_quantifier'],
                    "doc_id": row["doc_id"],
                    "n_tokens_context": n_tokens_context,
                    "context": context,
                    "losses": plosses["losses"] if plosses else None,
                    "verb_analysis": plosses['verb_analysis'] if plosses else None,
                    "acceptable": plosses['acceptable'] if (plosses and "acceptable" in plosses) else None
                }
            )

    
        # partial saves
        if i % 50 == 0:
            pd.DataFrame(results).to_csv(
                output_file, 
                index=False,
                # append instead of rewrite
                # mode='a' if os.path.exists(output_file) else 'w'
            )
            # results = []
    pd.DataFrame(results).to_csv(
        output_file, 
        index=False,
        # mode='a' if os.path.exists(output_file) else 'w'
    )





