import re
import torch
import pytest
import nltk
from torch.nn import CrossEntropyLoss
from nltk.tokenize import word_tokenize
from utils.lm import load_tokenizer, load_mistral

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

#### GET QUANTIFIED SENTENCES ####

def get_quantified_sentences(text, original_quantifier):
    generic = text.strip().capitalize()
    if original_quantifier != "gen":
        generic = re.sub(f"^{original_quantifier.capitalize()}", "", generic).strip()
    generic = generic[0].lower() + generic[1:]
    return {
        "gen": generic,
        "all": f"all {generic}",
        "some": f"some {generic}",
        "most": f"most {generic}",
    }

@pytest.mark.parametrize(
    "text,original_quantifier,result", 
    [
        ("tigers have stripes", "gen", {"gen": "tigers have stripes", "all": "all tigers have stripes", "most": "most tigers have stripes", "some": "some tigers have stripes"}),
        ("all tigers have stripes", "all", {"gen": "tigers have stripes", "all": "all tigers have stripes", "most": "most tigers have stripes", "some": "some tigers have stripes"}),
        ("most tigers have stripes", "most", {"gen": "tigers have stripes", "all": "all tigers have stripes", "most": "most tigers have stripes", "some": "some tigers have stripes"}),
        ("some tigers have stripes", "some", {"gen": "tigers have stripes", "all": "all tigers have stripes", "most": "most tigers have stripes", "some": "some tigers have stripes"})
    ]
)
def test_get_quantified_sentences(text, original_quantifier, result):
    assert get_quantified_sentences(text, original_quantifier) == result


#### VERB DETECTION ####   

def get_verb_token_position(sentence, tokenizer):
    """This function only works for very simple sentences, will mess on clauses etc, but it's enough for our purposes. Will match greedily the first verb it finds, return position -1 if it does not find any."""
    tags = nltk.pos_tag(word_tokenize(sentence))
    verb = [tag for tag in tags if tag[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]]
    if not verb:
        return {"position":-1, "token":""}
    verb = verb[0][0]
    
    tokenizer_tokens = tokenizer.convert_ids_to_tokens([t for t in tokenizer(sentence)["input_ids"]])
    min_distance = 100000
    min_distance_i = -1
    verb_token = ""
    for i,token in enumerate(tokenizer_tokens):
        # edit distance to verb
        distance = nltk.edit_distance(token.replace("▁", ""), verb)
        if distance == 0:
            min_distance_i = i
            min_distance = distance
            verb_token = token
            break
        elif distance < min_distance:
            min_distance_i = i
            min_distance = distance
            verb_token = token
    return {"position":min_distance_i, "token":verb_token}

@pytest.mark.parametrize(
    "sentence,result",
    [
        ("tigers have stripes", {"position":4, "token":"▁have"}),
        ("tigers are striped", {"position":4, "token":"▁are"}),
        ("stripes in tigers align on the sides", {"position":7, "token":"▁align"}),
        ("stripes and tigers together", {"position":-1, "token":""}),
        ("tigers eat rabbits and kill humans", {"position":4, "token":"▁eat"}),
    ]
)
def test_get_verb_token_position(sentence, result):
    tokenizer = load_tokenizer("mistralai/Mistral-7B-v0.1")
    assert get_verb_token_position(sentence, tokenizer) == result


#### LOSS CALCULATION ####

def get_losses(logits, 
    labels, 
    loss_fct = CrossEntropyLoss(reduction="mean"), 
    quantifiers=["gen", "some", "all", "most"]
):
    ppls = {}
    for q in quantifiers:
        lgts = logits[q]
        lbls = torch.tensor(labels[q])
        lbls.to(lgts.device)
        # print shapes of both
        # print(f"{q} logits: ", [lg[l].item() for lg,l in zip(lgts,lbls)])
        # sftmx = torch.nn.functional.softmax(lgts, dim=1)
        # print(f"{q} probs:", [lg[l].item() for lg,l in zip(sftmx,lbls)])
        ppls[q] = loss_fct(lgts, lbls).item()#(loss_fct(lgts, lbls).sum(0)/len(lbls)).item()
    return ppls


def test_get_losses():
    dummy_labels = {'gen': [4216, 4384], 'all': [4216, 4384], 'some': [4216, 4384], 'most': [4216, 4384]}
    dummy_logits = {
        'gen': torch.zeros(2, 32000),
        'all': torch.zeros(2, 32000),
        'some': torch.zeros(2, 32000),
        'most': torch.zeros(2, 32000)
    }
    for q in dummy_labels:
        dummy_logits[q][:, dummy_labels[q]] = 1.0

    # force gen > all > most > some
    dummy_logits["gen"]*=4
    dummy_logits["all"]*=3
    dummy_logits["most"]*=2
    dummy_logits["some"]*=1

    # the bigger the logits on the correct tokens, the lower the loss will be
    losses = get_losses(dummy_logits, dummy_labels)
    print(losses)
    assert losses["gen"] < losses["all"]
    assert losses["all"] < losses["most"]
    assert losses["most"] < losses["some"]

#### ACCEPTABILITY ####

def get_property_losses_from_context_quantified_sentences(
    quantified_sentences, 
    model, 
    tokenizer,
    context = "", 
    loss_fct = CrossEntropyLoss(reduction="mean"),
    max_property_tokens: int = None,
    full_sentence_loss = False,
    debug=False
):
    """
        The batch size of the forward pass is implictely the amount of sentences passed (4), sorry.
        Args:
            quantified_sentences (dict): Keys are quantifiers, values are respectively quantified variations of a generic. If the quantifiers are more than a token it will not work. Tested for ["gen", "some", "all", "most"].
            context (str): String containing the left-side context of the quantified sentences.
            model: Autoregressive LM. Only tested for Mistral.
            tokenizer: Just that, the tokenizer. 
            loss_fct: Loss function. Only tested for CrossEntropyLoss.
            max_property_tokens: Max tokens of the predicate component over which to compute the loss.
            full_sentence_loss: Perform acceptability calculation with full loss.
            debug: Print debug information.
    """
    quantifiers = quantified_sentences.keys()
    verb_analysis = get_verb_token_position(quantified_sentences["gen"], tokenizer)
    verb_token_position = verb_analysis["position"]
    if "gpt2" in model.name_or_path:
        # no <s>
        verb_token_position -= 1
    if verb_token_position <= 1:
        return None
        # verb_token_position = 1
    input_sentences = [f"{context} {quantified_sentences[q]}".strip() for q in quantifiers]
    inputs = tokenizer(input_sentences, return_tensors="pt", padding=True)
    inputs.to(model.device)
    if context != "":
        context_input = tokenizer(context, return_tensors="pt", padding=False)
        context_length_in_tokens = len(context_input["input_ids"][0])-1
    else:
        context_length_in_tokens = 0
    outputs = model(**inputs)
    labels = {}
    logits = {}
    for i, quantifier in enumerate(quantifiers):
        if quantifier == "gen":
            labels[quantifier] = inputs["input_ids"].tolist()[i][context_length_in_tokens+verb_token_position+1:-1]
            logits[quantifier] = outputs.logits[i][context_length_in_tokens+verb_token_position:-2].to("cpu")
        else:
            labels[quantifier] = inputs["input_ids"].tolist()[i][context_length_in_tokens+verb_token_position+2:]
            logits[quantifier] = outputs.logits[i][context_length_in_tokens+verb_token_position+1:-1].to("cpu")
        if full_sentence_loss:
            labels[quantifier] = inputs["input_ids"].tolist()[i][1:]
            logits[quantifier] = outputs.logits[i][:-1].to("cpu")
    if max_property_tokens and max_property_tokens > 0:
        for q in quantifiers:
            labels[q] = labels[q][:max_property_tokens]
            logits[q] = logits[q][:max_property_tokens]

    # try:
    results = {}
    results["quantified_sentences"] = quantified_sentences.copy()
    results["losses"] = get_losses(logits, labels, loss_fct, quantifiers)
    results["verb_analysis"] = verb_analysis
    min_loss = 100000
    for q in quantifiers:
        if results["losses"][q] < min_loss:
            min_loss = results["losses"][q]
            results["acceptable"] = q
    if debug:
        print("CONTEXT LOSS DEBUG -------------------")
        print(verb_analysis)
        tokens = [tokenizer.convert_ids_to_tokens(i) for i in inputs["input_ids"]]
        print(tokens)
        if not full_sentence_loss:
            print("gen labels: ", tokens[0][context_length_in_tokens+verb_token_position+1:-1])
            print("gen logits: ", tokens[0][context_length_in_tokens+verb_token_position:-2])
            print("all labels: ", tokens[2][context_length_in_tokens+verb_token_position+2:])
            print("all logits: ", tokens[2][context_length_in_tokens+verb_token_position+1:-1])
            print("most labels: ", tokens[1][context_length_in_tokens+verb_token_position+2:])
            print("most logits: ", tokens[1][context_length_in_tokens+verb_token_position+1:-1])
            print("some labels: ", tokens[3][context_length_in_tokens+verb_token_position+2:])
            print("some logits: ", tokens[3][context_length_in_tokens+verb_token_position+1:-1])
        else:
            print("gen labels: ", tokens[0][1:])
            print("gen logits: ", tokens[0][:-1])
            print("all labels: ", tokens[2][1:])
            print("all logits: ", tokens[2][:-1])
            print("most labels: ", tokens[1][1:])
            print("most logits: ", tokens[1][:-1])
            print("some labels: ", tokens[3][1:])
            print("some logits: ", tokens[3][:-1])
        print("losses: ", results["losses"])
        print("pl-acceptable: ", results["acceptable"])
        print("labels: ", labels)
        print("input_ids: ", inputs["input_ids"].tolist())
        # print("logits: ", logits)
        print("shape logits: ", logits["gen"].shape)
        print("-----------------------------------")
    return results

@pytest.fixture(scope="module")
def load_test_mistral():
    model_id = "mistralai/Mistral-7B-v0.1"
    model = load_mistral(model_id)
    tokenizer = load_tokenizer(model_id)
    return (model, tokenizer)

@pytest.mark.parametrize(
    "quantified_sentences,context,acceptable",
    [
        # quantization can actually flip some of these results
        (get_quantified_sentences("tigers have stripes", "gen"), "", "all"),
        (get_quantified_sentences("tigers live in zoos", "gen"), "", "some"),
        (get_quantified_sentences("all tigers have stripes", "all"), "", "all"),
        (get_quantified_sentences("most tigers live in zoos", "most"), "", "some"),
        (get_quantified_sentences("tigers eat humans", "gen"), "", "all"),
        (get_quantified_sentences("all tigers live in the jungle", "all"), "", "all"),
        (get_quantified_sentences("most tigers live in zoos", "most"), "Every single tiger in the world is captive in a zoo.", "all"),

    ]
)
def test_get_property_losses_from_context_quantified_sentences(
    quantified_sentences, context, acceptable, load_test_mistral
):
    results = get_property_losses_from_context_quantified_sentences(
        quantified_sentences,
        load_test_mistral[0],
        load_test_mistral[1],
        context,
        debug=True
    )
    assert results["acceptable"] == acceptable
