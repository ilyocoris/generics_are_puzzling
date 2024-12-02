"""
Acceptability of GEN, ALL, MOST, SOME for sentences in a cleaned version of GenericsKB.
"""
# CUDA_VISIBLE_DEVICES=5,6,7 python -m run_gkb
import pandas as pd
from tqdm import tqdm
from utils.lm import load_tokenizer, load_mistral
from utils.acceptability import get_quantified_sentences, get_property_losses_from_context_quantified_sentences

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

model_nicknames = ["8x7B"] #"8x7B",  "8x7B-Instruct", "7B-Instruct",

data = pd.read_csv("data/GenericsKB_bare_plurals/gkb_clean.csv")


for model_nickname in model_nicknames:
    print(f"Running model {model_nickname}")
    model = load_mistral(model_nickname)
    tokenizer = load_tokenizer(model_nickname)
    output_file = f"results/gkb_acceptability_{model_nickname}.csv"

    # add 3 columns to data: losses, verb, pl_acceptable
    data['losses'] = None
    data['verb'] = None
    data['acceptable'] = None

    results = []

    for i, row in tqdm(data.iterrows(), total=len(data), desc=f"Model {model_nickname}"):
        quantified_sentences = get_quantified_sentences(row['text'], row['original_quantifier'])
        plosses = get_property_losses_from_context_quantified_sentences(
            quantified_sentences = quantified_sentences, 
            model = model, 
            tokenizer = tokenizer,
            context = "",
            # max_property_tokens = 5
        )

        results.append(
            {
                "text": row['text'],
                "original_quantifier": row['original_quantifier'],
                "v_lemma": row['v_lemma'],
                "losses": plosses["losses"] if plosses else None,
                "verb_analysis": plosses['verb_analysis'] if plosses else None,
                "acceptable": plosses['acceptable'] if plosses else None
            }
        )

        # partial saves
        if i % 1000 == 0:
            pd.DataFrame(results).to_csv(output_file, index=False)
    pd.DataFrame(results).to_csv(output_file, index=False)





