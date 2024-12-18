## Generics are Puzzling. Can Language Models find the missing piece?

This repository contains the code and data used for the paper [Generics are puzzling. Can language models find the missing piece?](https://arxiv.org/abs/2412.11318).

[Here](https://gustavocilleruelo.com/generics_are_puzzling) you can find an informal write-up on the paper & what it is trying to accomplish.

The most important data is ConGen, a collection of naturally occurring generic and quantified sentences with their corresponding context. You can download the .csv in `data/congen` and open them in any Excel-like program or load them to python by doing:

```
import pandas as pd
sentences = pd.read_csv('data/congen/sentences.csv')
documents = pd.read_csv('data/congen/documents.csv')
```

Sentences are just the generics, and documents are the contexts (both are matched through `doc_id`).

### Data

The `data` folder contains the three data collections used in the experiments: ConGen, a curated subset of GenericsKB and a collection of synthetic stereotypes.

### Code

The `scripts` folder contains the scripts and utility functions used to run the experiments in the paper.
