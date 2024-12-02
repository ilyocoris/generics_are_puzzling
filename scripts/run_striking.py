"""
Acceptability of GEN, ALL, MOST, SOME for sentences in a collection of striking generics.
"""
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m run_striking
import pandas as pd
from tqdm import tqdm
from utils.lm import load_tokenizer, load_mistral
from utils.acceptability import get_quantified_sentences, get_property_losses_from_context_quantified_sentences


model_nicknames = ["8x7B", "8x7B-Instruct"] # ["7B", "7B-Instruct"] #["8x7B",  "8x7B-Instruct"] #"8x7B",  "8x7B-Instruct", "7B-Instruct",

data = pd.read_csv("data/striking/stereotypes.csv")


for model_nickname in model_nicknames:
    print(f"Running model {model_nickname}")
    model = load_mistral(model_nickname)
    tokenizer = load_tokenizer(model_nickname)
    output_file = f"results/striking_acceptability_{model_nickname}.csv"

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
        )

        results.append(
            {
                "text": row['text'],
                "original_quantifier": row['original_quantifier'],
                "mood": row['mood'],
                "target": row['target'],
                "sentence_type": row['sentence_type'],
                "singular": row['singular'],
                "plural": row['plural'],
                "losses": plosses["losses"] if plosses and "losses" in plosses else None,
                "verb_analysis": plosses['verb_analysis'] if plosses and "verb_analysis" in plosses else None,
                "acceptable": plosses['acceptable'] if plosses and "acceptable" in plosses else None,
                "source": row['source'],
            }
        )

        # partial saves
        if i % 100 == 0:
            pd.DataFrame(results).to_csv(output_file, index=False)
    pd.DataFrame(results).to_csv(output_file, index=False)





