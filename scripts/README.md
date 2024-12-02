This folder contains the scripts for the experiments in the paper.

The most important function is in `utils/acceptability.py` and is `get_property_losses_from_context_quantified_sentences` which gets called to compute the surprisal on the property tokens. Then the `run_....py` scripts use this function to calculate the p-acceptabilities across the different datasets.
