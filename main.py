from corpus import load_brown, split_brown, load_spanish
from language_model import TrigramLanguageModel
from segmentation import ViterbiSegmenter
from pos_tagger import TrigramPOSTagger, MorphologyAwarePOSTagger
from baselines import GreedyLongestMatchSegmenter, MostFrequentTagger
from results import segmentation_accuracy, spanish_segmentation_accuracy, pos_accuracy

print("\n====================")
print("ENGLISH")
print("====================")

brown = load_brown()
english_train, english_test = split_brown(brown)

# ---------- Language model ----------

english_lm = TrigramLanguageModel()

english_lm.train([
    [word for word, tag in sentence]
    for sentence in english_train
])

viterbi_segmenter = ViterbiSegmenter(english_lm)

greedy_segmenter = GreedyLongestMatchSegmenter(
    english_lm.vocabulary
)

# ---------- Segmentation ----------

print("\nEnglish segmentation...")

english_viterbi_seg_acc = segmentation_accuracy(
    english_test,
    viterbi_segmenter
)

english_greedy_seg_acc = segmentation_accuracy(
    english_test,
    greedy_segmenter
)

print(
    f"Viterbi segmentation accuracy: "
    f"{english_viterbi_seg_acc:.4%}"
)

print(
    f"Greedy segmentation accuracy: "
    f"{english_greedy_seg_acc:.4%}"
)

print(
    f"Improvement: "
    f"{english_viterbi_seg_acc - english_greedy_seg_acc:.4%}"
)


# ---------- POS ----------

english_tagger = TrigramPOSTagger()
english_tagger.train(english_train)

english_baseline = MostFrequentTagger()
english_baseline.train(english_train)

english_predictions = []
english_baseline_predictions = []

print("\nEnglish POS tagging...")

for i, sentence in enumerate(english_test):

    words = [
        word
        for word, tag in sentence
    ]

    english_predictions.append(
        english_tagger.tag(words)
    )

    english_baseline_predictions.append(
        english_baseline.tag(words)
    )

    if (i + 1) % 1000 == 0:
        print(f"  {i + 1}/{len(english_test)}")


english_pos_acc = pos_accuracy(
    english_test,
    english_predictions
)

english_baseline_acc = pos_accuracy(
    english_test,
    english_baseline_predictions
)

print(
    f"\nViterbi POS accuracy: "
    f"{english_pos_acc:.4%}"
)

print(
    f"Most-frequent-tag accuracy: "
    f"{english_baseline_acc:.4%}"
)

print(
    f"Improvement: "
    f"{english_pos_acc - english_baseline_acc:.4%}"
)


# ============================================================
# SPANISH
# ============================================================

print("\n====================")
print("SPANISH")
print("====================")

spanish_train, spanish_dev, spanish_test = load_spanish()

# ---------- Language model ----------

spanish_lm = TrigramLanguageModel()

spanish_lm.train([
    [token["form"] for token in sentence]
    for sentence in spanish_train
])

spanish_viterbi_segmenter = ViterbiSegmenter(
    spanish_lm
)

spanish_greedy_segmenter = GreedyLongestMatchSegmenter(
    spanish_lm.vocabulary
)

# ---------- Segmentation ----------

print("\nSpanish segmentation...")

spanish_viterbi_seg_acc = spanish_segmentation_accuracy(
    spanish_test,
    spanish_viterbi_segmenter
)

spanish_greedy_seg_acc = spanish_segmentation_accuracy(
    spanish_test,
    spanish_greedy_segmenter
)

print(
    f"Viterbi segmentation accuracy: "
    f"{spanish_viterbi_seg_acc:.4%}"
)

print(
    f"Greedy segmentation accuracy: "
    f"{spanish_greedy_seg_acc:.4%}"
)

print(
    f"Improvement: "
    f"{spanish_viterbi_seg_acc - spanish_greedy_seg_acc:.4%}"
)


# ============================================================
# Spanish POS
# ============================================================

# ---------- Plain POS ----------

spanish_plain_train = [
    [
        (token["form"], token["upos"])
        for token in sentence
    ]
    for sentence in spanish_train
]

spanish_plain_test = [
    [
        (token["form"], token["upos"])
        for token in sentence
    ]
    for sentence in spanish_test
]

spanish_tagger = TrigramPOSTagger()
spanish_tagger.train(spanish_plain_train)

spanish_baseline = MostFrequentTagger()
spanish_baseline.train(spanish_plain_train)

spanish_predictions = []
spanish_baseline_predictions = []

print("\nSpanish plain POS tagging...")

for sentence in spanish_plain_test:

    words = [
        word
        for word, tag in sentence
    ]

    spanish_predictions.append(
        spanish_tagger.tag(words)
    )

    spanish_baseline_predictions.append(
        spanish_baseline.tag(words)
    )

spanish_pos_acc = pos_accuracy(
    spanish_plain_test,
    spanish_predictions
)

spanish_baseline_acc = pos_accuracy(
    spanish_plain_test,
    spanish_baseline_predictions
)

print(
    f"Plain Viterbi POS accuracy: "
    f"{spanish_pos_acc:.4%}"
)

print(
    f"Most-frequent-tag accuracy: "
    f"{spanish_baseline_acc:.4%}"
)

print(
    f"Improvement: "
    f"{spanish_pos_acc - spanish_baseline_acc:.4%}"
)


# ============================================================
# Spanish morphology-aware POS
# ============================================================

spanish_morph_tagger = MorphologyAwarePOSTagger()

spanish_morph_tagger.train_spanish(
    spanish_train
)

spanish_morph_gold = []
spanish_morph_predictions = []

print("\nSpanish morphology-aware POS tagging...")

for sentence in spanish_test:

    gold_sentence = []

    words = []

    for token in sentence:

        words.append(token["form"])

        morph_tag = (
            MorphologyAwarePOSTagger.make_morph_tag(
                token["upos"],
                token["feats"]
            )
        )

        gold_sentence.append(
            (token["form"], morph_tag)
        )

    spanish_morph_gold.append(gold_sentence)

    spanish_morph_predictions.append(
        spanish_morph_tagger.tag(words)
    )

spanish_morph_acc = pos_accuracy(
    spanish_morph_gold,
    spanish_morph_predictions
)

print(
    f"Morphology-aware POS accuracy: "
    f"{spanish_morph_acc:.4%}"
)

print(
    f"Difference from plain POS: "
    f"{spanish_morph_acc - spanish_pos_acc:.4%}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================")
print("FINAL RESULTS")
print("====================")

print("\nSegmentation:")
print(
    f"English Viterbi:  {english_viterbi_seg_acc:.4%}"
)
print(
    f"English Greedy:   {english_greedy_seg_acc:.4%}"
)
print(
    f"Spanish Viterbi:  {spanish_viterbi_seg_acc:.4%}"
)
print(
    f"Spanish Greedy:   {spanish_greedy_seg_acc:.4%}"
)

print("\nPOS:")
print(
    f"English Viterbi:  {english_pos_acc:.4%}"
)
print(
    f"English Baseline: {english_baseline_acc:.4%}"
)
print(
    f"Spanish Viterbi:  {spanish_pos_acc:.4%}"
)
print(
    f"Spanish Baseline: {spanish_baseline_acc:.4%}"
)
print(
    f"Spanish Morph:    {spanish_morph_acc:.4%}"
)

# -------------------------
# Sample Sentences
# -------------------------


print("#"*20 + "SAMPLE SENTENCES" + "#"*20)
# -------------------------
# English
# -------------------------

brown = load_brown()
english_train, _ = split_brown(brown)

english_lm = TrigramLanguageModel()

english_words = [
    [word for word, tag in sentence]
    for sentence in english_train
]

english_lm.train(english_words)

english_segmenter = ViterbiSegmenter(english_lm)

english_tagger = TrigramPOSTagger()
english_tagger.train(english_train)


# -------------------------
# Spanish
# -------------------------

spanish_train, _, _ = load_spanish()

spanish_lm = TrigramLanguageModel()

spanish_words = [
    [token["form"] for token in sentence]
    for sentence in spanish_train
]

spanish_lm.train(spanish_words)

spanish_segmenter = ViterbiSegmenter(spanish_lm)

spanish_tagger = TrigramPOSTagger()

spanish_plain_sentences = [
    [
        (token["form"], token["upos"])
        for token in sentence
    ]
    for sentence in spanish_train
]

spanish_tagger.train(spanish_plain_sentences)

spanish_morph_tagger = MorphologyAwarePOSTagger()
spanish_morph_tagger.train_spanish(spanish_train)


# -------------------------
# Sample sentences
# -------------------------

english_sample_1 = "thequickbrownfoxjumpsoverthelazydog"

english_sample_2 = "tobeornottobethatisthequestion"

spanish_sample_1 = "mispadrespuedenviajar"

spanish_sample_2 = "elcielodespejadoesazul"

spanish_sample_3 = "maríaleeunlibro"


# -------------------------
# English output
# -------------------------

print("\n=== ENGLISH ===")

for sample in [
    english_sample_1,
    english_sample_2,
]:

    segmented = english_segmenter.segment(sample)

    print("\nInput:")
    print(sample)

    print("\nSegmentation:")
    print(segmented)

    print("\nPOS:")
    print(english_tagger.tag(segmented))


# -------------------------
# Spanish output
# -------------------------

print("\n=== SPANISH ===")

for sample in [
    spanish_sample_1,
    spanish_sample_2,
    spanish_sample_3
]:

    segmented = spanish_segmenter.segment(sample)

    print("\nInput:")
    print(sample)

    print("\nSegmentation:")
    print(segmented)

    print("\nPOS:")
    print(spanish_tagger.tag(segmented))

    print("\nMorphology-aware POS:")
    print(spanish_morph_tagger.tag(segmented))
