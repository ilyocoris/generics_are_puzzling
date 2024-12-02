import re
import pytest
import pandas as pd
from typing import List
from utils.lm import load_tokenizer
from sentence_splitter import split_text_into_sentences

def load_congen(congen_folder_path:str = "./data/congen"):
    sentences = pd.read_csv(f"{congen_folder_path}/sentences.csv")
    documents = pd.read_csv(f"{congen_folder_path}/documents.csv")
    return sentences, documents

def find_document_by_id(documents, doc_id):
    return documents[documents["doc_id"] == doc_id].iloc[0]

def get_sentence_index_in_split_document(text, doc_split):
    for i, s in enumerate(doc_split):
        if re.match(rf"{re.escape(text)}", s):
            return i
    return -1

def split_doc_into_dense_sentences(doc:str):
    doc_split = split_text_into_sentences(doc, language="en")
    doc_split = [s for s in doc_split if s != ""]
    return doc_split

@pytest.mark.parametrize(
    "sentence,doc_split,result",
    [
        ("This is a simple document.", split_doc_into_dense_sentences("This is a simple document. It has two sentences."), 0),
        ("It has two sentences.", split_doc_into_dense_sentences("This is a simple document. It has two sentences."), 1),
        ("It has three sentences.", split_doc_into_dense_sentences("This is a simple document. It has two sentences."), -1),
        ("It has two", split_doc_into_dense_sentences("This is a simple document. It has two sentences."), 1),
        ("Tigers are white.", split_doc_into_dense_sentences("The year is 2024. Earth is covered in snow. Tigers are white. Elephants are pink."), 2),
    ]
)
def test_get_sentence_index_in_split_document(sentence, doc_split, result):
    assert get_sentence_index_in_split_document(sentence, doc_split) == result

def get_left_contexts(
    context: str, 
    sentence: str, 
    splits: List[int], 
    context_type: str, 
    tokenizer=None
):
    """
        Args:
            context (str): The context to extract the sentences from.
            sentence (str): Sentence to build context for.
            splits (list[int]): Contexts that are wanted, for example [1,2,3,4,5] + "sentence" will be contexts of 1, 2, 3... sentences before the sentence. If [10, 20, 30, 60, 100] will be the sentence with the 10, 20, 30, ... tokens to the left. This is done up to finishing the context, so it may return shorter lists if the context is small.
            context_type (str): "sentences" or "tokens"
            tokenizer: Tokenizer to use if modality is "token"
        Returns:
            A dictionary with the number of sentences/tokens and the appropiate context.
    """
    doc_split = split_doc_into_dense_sentences(context)
    sentence_index = get_sentence_index_in_split_document(sentence, doc_split)
    if sentence_index == -1:
        print(f"Sentence {sentence} not found in context.")
        return None
    results = {}
    if context_type == "sentences":
        for i in splits:
            if i == 0:
                results[i] = ""
                continue
            if sentence_index-i < 0:
                print(f"Index {sentence_index} is out of bounds.")
                break
            results[i] = " ".join(doc_split[sentence_index-i:sentence_index])
    elif context_type == "tokens":
        left_context = " ".join(doc_split[:sentence_index])
        tokenized_left_context = tokenizer.tokenize(left_context)
        for i in splits:
            if i == 0:
                results[i] = ""
                continue
            if i > len(tokenized_left_context):
                break
            partial_left_context = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(
                    tokenized_left_context[-i:]
                )
            )
            results[i] = partial_left_context
    return results

@pytest.fixture(scope="module")
def load_test_tokenizer():
    return load_tokenizer("7B")


@pytest.mark.parametrize(
    "context,sentence,splits,context_type,result",
    [
        ("The year is 2024. Earth is covered in snow. Tigers are white. Elephants are pink.", "Tigers are white.", [0, 6, 8, 11, 1000], "tokens", {
            0: "",
            6: "Earth is covered in snow.",
            8: "4. Earth is covered in snow.",
            11: "2024. Earth is covered in snow."
        }),
        ("The year is 2024. Earth is covered in snow. Tigers are white. Elephants are pink.", "Tigers are white.", [0, 1, 2, 1000], "sentences", {
            0: "",
            1: "Earth is covered in snow.",
            2: "The year is 2024. Earth is covered in snow."
        }),
    ]
)
def test_get_left_contexts(context, sentence, splits, context_type, result, load_test_tokenizer):
    assert get_left_contexts(context, sentence, splits, context_type, load_test_tokenizer) == result

