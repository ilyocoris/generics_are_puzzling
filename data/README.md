This folder contains the data used in the paper.

**ConGen**: This dataset is split in two files: sentences.csv contains the bare plural generics and quantified sentences with an id pointing to the original multi-sentence context and the documents.csv file contains the context documents with their respective ids.

```
ConGen/
    sentences.csv
    documents.csv
```

**GenericsKB_bare_plurals**: This is a subset of sentences in [GenericsKB](https://arxiv.org/abs/2005.00660) with some additional syntactic information extracted with [stanza](https://stanfordnlp.github.io/stanza/) to select bare plurals.

```
GenericsKB_bare_plurals/
    gkb_clean.csv
```

**Stereotypes**: We attach the collection of real/invented and positive/negative stereotypes we use. Note that in this csv the 3 types of sentence we study are already constructed, so there are 3 entries for each plural-predicate pair.

```
striking/
    stereotypes.csv
```
