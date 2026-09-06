# Question 1: Word Segmentation and POS Tagging

This project implements word segmentation and Part-of-Speech (POS) tagging for **English and Spanish** using trigram language models and Viterbi dynamic programming.

## Overview

The system performs the following tasks:

1. Word segmentation from text with spaces removed.
2. POS tagging of the segmented words.
3. Morphology-aware POS tagging for Spanish using gender and number features.
4. Comparison against simple baseline methods.
5. Evaluation using accuracy, confusion matrices, and error-source analysis.

The implementation uses:

- **English:** NLTK Brown Corpus
- **Spanish:** Universal Dependencies Spanish-GSD Corpus

---

## Project Structure

```text
q1/
├── corpus.py
├── language_model.py
├── segmentation.py
├── pos_tagger.py
├── baselines.py
├── evaluation.py
├── results.py
├── main.py
├── sample_outputs.py
└── data/
    └── UD_Spanish-GSD/
```

### Files

#### `corpus.py`

Handles loading and preprocessing of the corpora.

- Loads the NLTK Brown Corpus.
- Creates an 80/20 training/testing split for English.
- Loads the Spanish UD-GSD train, development, and test sets.
- Parses CoNLL-U formatted data.

#### `language_model.py`

Implements a **trigram word language model**.

The model estimates:

```text
P(w3 | w1, w2)
```

using trigram and bigram counts with add-one smoothing.

It also provides log probabilities for use by the Viterbi segmenter.

#### `segmentation.py`

Implements Viterbi-based word segmentation.

Given an input such as:

```text
thequickbrownfox
```

the algorithm searches over possible word boundaries and selects the segmentation with the highest trigram language-model probability.

Example:

```text
thequickbrownfox
        ↓
the quick brown fox
```

#### `pos_tagger.py`

Contains two POS taggers:

- `TrigramPOSTagger`
- `MorphologyAwarePOSTagger`

The standard POS tagger uses:

- Emission probabilities `P(word | tag)`
- Trigram transition probabilities `P(tag3 | tag1, tag2)`
- Viterbi decoding

The morphology-aware Spanish tagger extends POS tags using gender and number.

For example:

```text
NOUN-Fem-Sg
ADJ-Fem-Pl
DET-Masc-Sg
```

#### `baselines.py`

Contains the two baseline systems required for comparison.

**Greedy Longest-Match Segmentation**

At every position, the longest word in the vocabulary is selected.

**Most-Frequent-Tag POS Tagging**

Each word is assigned the POS tag that occurred most frequently with that word in the training data.

#### `evaluation.py`

Provides evaluation utilities including:

- POS accuracy
- Confusion matrix
- Segmentation error-source analysis

The error-source analysis distinguishes between:

1. Errors caused by incorrect segmentation.
2. Errors where segmentation is correct but the POS tag is incorrect.

#### `results.py`

Runs the complete evaluation pipeline and compares the proposed models against the baselines.

---

# Requirements

The project was developed and tested using:

- **Python 3.11**
- NLTK

Install the required dependency:

```bash
pip install nltk
```

The first run downloads the Brown Corpus automatically.

---

# Spanish Corpus

The Spanish model uses the **UD Spanish-GSD** corpus.

Place the corpus at:

```text
data/spanish/UD_Spanish-GSD/
```

The expected files are:

```text
es_gsd-ud-train.conllu
es_gsd-ud-dev.conllu
es_gsd-ud-test.conllu
```

The corpus directory is excluded from Git using `.gitignore`.

---

# Running the Project

Run the main evaluation script:

```bash
python results.py
```

This evaluates:

- English Viterbi segmentation
- English greedy segmentation
- English Viterbi POS tagging
- English most-frequent-tag baseline
- Spanish Viterbi segmentation
- Spanish greedy segmentation
- Spanish plain POS tagging
- Spanish most-frequent-tag baseline
- Spanish morphology-aware POS tagging

To run the sample demonstrations:

```bash
python sample_outputs.py
```

---

# Using the Implementation in Other Modules

The components are designed to be imported independently. Other group members can therefore use the implementation directly without copying the classes into their own files.

If the files are in the same project directory, imports can be made directly:

```python
from language_model import TrigramLanguageModel
from segmentation import ViterbiSegmenter
from pos_tagger import TrigramPOSTagger
```

## 1. Word Segmentation

Train the trigram language model on tokenized training sentences:

```python
lm = TrigramLanguageModel()
lm.train(training_sentences)
```

Create the Viterbi segmenter:

```python
segmenter = ViterbiSegmenter(lm)
```

Then segment text with spaces removed:

```python
text = "thequickbrownfox"

words = segmenter.segment(text)

print(words)
```

Example output:

```text
['the', 'quick', 'brown', 'fox']
```

The same interface can be used for Spanish by training the language model on the Spanish training sentences.

## 2. POS Tagging

Import and train the trigram POS tagger:

```python
from pos_tagger import TrigramPOSTagger

tagger = TrigramPOSTagger()
tagger.train(training_tagged_sentences)
```

The English tagger expects training sentences in the form:

```python
[
    [
        ("the", "AT"),
        ("quick", "JJ"),
        ("fox", "NN")
    ]
]
```

Tag an already-segmented sentence:

```python
words = ["the", "quick", "brown", "fox"]

tagged_words = tagger.tag(words)

print(tagged_words)
```

Example:

```text
[
    ('the', 'AT'),
    ('quick', 'JJ'),
    ('brown', 'JJ'),
    ('fox', 'NN')
]
```

The POS tagger should receive segmented words. In the complete pipeline, segmentation is performed first and the resulting word list is passed to the POS tagger.

## 3. Complete Segmentation + POS Pipeline

A complete pipeline can be constructed as follows:

```python
from language_model import TrigramLanguageModel
from segmentation import ViterbiSegmenter
from pos_tagger import TrigramPOSTagger

# Train word language model
lm = TrigramLanguageModel()
lm.train(training_sentences)

# Create segmenter
segmenter = ViterbiSegmenter(lm)

# Train POS tagger
tagger = TrigramPOSTagger()
tagger.train(training_tagged_sentences)

# Input with spaces removed
text = "thequickbrownfox"

# Step 1: segmentation
words = segmenter.segment(text)

# Step 2: POS tagging
tagged_words = tagger.tag(words)

print(tagged_words)
```

The pipeline is:

```text
Input without spaces
        ↓
Trigram language model
        ↓
Viterbi word segmentation
        ↓
Segmented words
        ↓
Trigram POS tagger
        ↓
POS-tagged words
```

## 4. Spanish POS Tagging

The standard Spanish POS tagger uses Universal POS tags such as:

```text
DET
NOUN
ADJ
VERB
AUX
PRON
ADP
ADV
```

The loaded Spanish corpus stores additional information in each token, so it can be converted to `(word, UPOS)` pairs before training:

```python
plain_sentences = []

for sentence in spanish_train:
    converted_sentence = []

    for token in sentence:
        converted_sentence.append(
            (token["form"], token["upos"])
        )

    plain_sentences.append(converted_sentence)
```

Train the tagger:

```python
tagger = TrigramPOSTagger()
tagger.train(plain_sentences)
```

Then tag a sentence:

```python
words = ["la", "casa", "roja"]

print(tagger.tag(words))
```

## 5. Spanish Morphology-Aware POS Tagging

Import the morphology-aware tagger:

```python
from pos_tagger import MorphologyAwarePOSTagger
```

Train it directly using the parsed Spanish corpus:

```python
morph_tagger = MorphologyAwarePOSTagger()
morph_tagger.train_spanish(spanish_train)
```

Then:

```python
words = ["la", "casa", "roja"]

tagged_words = morph_tagger.tag(words)

print(tagged_words)
```

Morphological tags may look like:

```text
la       -> DET-Fem-Sg
casa     -> NOUN-Fem-Sg
roja     -> ADJ-Fem-Sg
```

---

# API Summary for Group Integration

| Component | Import | Main methods |
|---|---|---|
| Trigram language model | `TrigramLanguageModel` | `train()`, `score_sentence()` |
| Viterbi segmenter | `ViterbiSegmenter` | `segment()` |
| Trigram POS tagger | `TrigramPOSTagger` | `train()`, `tag()` |
| Morphology-aware tagger | `MorphologyAwarePOSTagger` | `train_spanish()`, `tag()` |
| Greedy segmenter | `GreedyLongestMatchSegmenter` | `segment()` |
| Most-frequent tagger | `MostFrequentTagger` | `train()`, `tag()` |

### Minimal Integration Example

For another module that needs segmentation followed by POS tagging:

```python
from language_model import TrigramLanguageModel
from segmentation import ViterbiSegmenter
from pos_tagger import TrigramPOSTagger

lm = TrigramLanguageModel()
lm.train(training_sentences)

segmenter = ViterbiSegmenter(lm)
words = segmenter.segment(input_text)

tagger = TrigramPOSTagger()
tagger.train(training_tagged_sentences)

result = tagger.tag(words)
```

The modules do not require `results.py`, `main.py`, or the evaluation code to be imported. Those scripts are intended for running the complete experiments and evaluation.

For straightforward integration, group members should keep their code in the same project directory and import the required classes rather than duplicating their implementations.

---

# Results

The current implementation produced the following results.

## Segmentation

| Language | Viterbi | Greedy Baseline | Improvement |
|---|---:|---:|---:|
| English | 48.8490% | 25.4970% | **+23.3519 pp** |
| Spanish | 14.9883% | 6.0890% | **+8.8993 pp** |

The Viterbi segmenter substantially outperformed the greedy longest-match baseline for both languages.

The improvement was particularly large for English, where Viterbi segmentation was 23.35 percentage points better than the baseline.

## POS Tagging

| Language | Viterbi | Most-Frequent Baseline | Improvement |
|---|---:|---:|---:|
| English | 92.9390% | 87.2853% | **+5.6537 pp** |
| Spanish | 92.9417% | 88.5500% | **+4.3917 pp** |

The trigram Viterbi tagger outperformed the most-frequent-tag baseline for both languages.

## Spanish Morphology-Aware Tagging

| Model | Accuracy |
|---|---:|
| Plain Viterbi POS | 92.9417% |
| Morphology-aware POS | 90.5667% |
| Difference | **-2.3750 pp** |

The morphology-aware model performed worse than the plain POS tagger in this experiment.

This suggests that explicitly adding gender and number introduced additional sparsity into the tag space. The additional morphological information therefore did not translate into better tagging accuracy with the current model and training data.

---

# Error Analysis

For the English end-to-end segmentation and POS-tagging pipeline:

```text
Total gold words:              181546
Segmentation-caused errors:     16305
Genuine tagging errors:          8405
```

Of the identified errors:

```text
Segmentation-associated: 65.99%
Genuine tagging:          34.01%
```

Therefore, segmentation was the larger source of errors in the complete pipeline, accounting for approximately two-thirds of the identified errors.

The error-source percentages refer to the errors identified by the complete segmentation-plus-tagging pipeline.

---

# Brown Corpus POS Tags

The English POS tagger uses the **Brown Corpus tagset**, rather than the Universal POS tagset.

Some examples include:

| Tag | Meaning | Example |
|---|---|---|
| `AT` | Article | `the` |
| `NN` | Common noun, singular | `question` |
| `NNS` | Common noun, plural | `questions` |
| `JJ` | Adjective | `quick` |
| `RB` | Adverb | `quickly` |
| `IN` | Preposition/subordinating conjunction | `in` |
| `CC` | Coordinating conjunction | `or` |
| `TO` | `to` | `to` |
| `BE` | Base form of *be* | `be` |
| `BEZ` | Present-tense singular form of *be* | `is` |
| `NP` | Proper noun | proper names |
| `PPSS` | Personal pronoun | `I`, `we`, `you` |
| `PP$` | Possessive personal pronoun | `my`, `your`, `his` |

For example:

```text
to/TO be/BE or/CC
```

and:

```text
is/BEZ
```

---

# Sample Output

The sample-output script demonstrates:

- English segmentation and POS tagging
- Spanish segmentation and POS tagging
- Spanish morphology-aware POS tagging

Run:

```bash
python sample_outputs.py
```

The predicted tags use the tagset of the corresponding training corpus.

---

# Method

## Word Segmentation

For each input string without spaces, possible vocabulary words beginning at each character position are considered.

The Viterbi algorithm maintains the best score for states containing the previous two words.

For a candidate word `w3`, the score is updated using:

```text
score + log P(w3 | w1, w2)
```

The highest-scoring complete segmentation is selected.

## POS Tagging

The POS tagger uses a trigram model over POS tags.

For each candidate tag:

```text
score =
    previous score
    + log P(tag | previous two tags)
    + log P(word | tag)
```

Viterbi decoding is then used to find the highest-scoring sequence of POS tags.

## Morphology-Aware Spanish Tagging

Spanish POS tags are extended using morphological features available in the UD corpus.

For example:

```text
NOUN + Feminine + Singular
```

becomes:

```text
NOUN-Fem-Sg
```

and:

```text
ADJ + Feminine + Plural
```

becomes:

```text
ADJ-Fem-Pl
```

This allows the tagger to represent grammatical agreement explicitly.

---

# Baselines

Two simple baselines are used.

### Greedy Longest-Match Segmenter

At every character position, the longest vocabulary item matching the remaining input is selected.

It does not consider future words or trigram context.

### Most-Frequent-Tag Tagger

For every word, the tag observed most frequently in the training data is selected.

It does not consider neighboring words or tag transitions.

These baselines provide a simple reference point for evaluating whether the language-model and Viterbi approaches provide meaningful improvements.

---

# Summary

The results show three main findings:

1. **Viterbi segmentation consistently outperformed greedy segmentation**, with particularly strong improvement for English.
2. **Trigram Viterbi POS tagging outperformed the most-frequent-tag baseline** for both English and Spanish.
3. **Morphology-aware Spanish tagging did not improve accuracy** in the current experiment. Instead, accuracy decreased by 2.38 percentage points.

The main weakness of the complete system is segmentation, particularly for Spanish. The error analysis also shows that segmentation mistakes account for a larger share of the end-to-end errors than genuine POS-tagging mistakes.
