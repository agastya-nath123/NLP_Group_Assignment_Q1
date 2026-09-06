from collections import Counter
import math


class TrigramLanguageModel:
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

        self.vocabulary = set()

        self.total_words = 0

    def train(self, sentences):
        """
        Train unigram, bigram, and trigram counts.

        sentences:
            List of sentences.
            Each sentence is a list of words.
        """

        for sentence in sentences:

            # Add sentence boundary markers.
            words = ["<START>", "<START>"] + [word.lower() for word in sentence] + ["<END>"]

            for word in sentence:
                word = word.lower()
                self.unigram_counts[word] += 1
                self.vocabulary.add(word)
                self.total_words += 1

            for i in range(2, len(words)):
                w1 = words[i - 2]
                w2 = words[i - 1]
                w3 = words[i]

                self.bigram_counts[(w1, w2)] += 1
                self.trigram_counts[(w1, w2, w3)] += 1

    def unigram_probability(self, word):
        """
        P(word)
        """

        if self.total_words == 0:
            return 0.0

        return self.unigram_counts[word] / self.total_words

    def trigram_probability(self, w1, w2, w3):
        """
        P(w3 | w1, w2)

        Uses add-one smoothing so unseen trigrams
        don't receive probability zero.
        """

        numerator = self.trigram_counts[(w1, w2, w3)] + 1

        denominator = self.bigram_counts[(w1, w2)] + len(self.vocabulary)

        return numerator / denominator

    def log_trigram_probability(self, w1, w2, w3):
        """
        Return log P(w3 | w1, w2).
        """

        probability = self.trigram_probability(w1, w2, w3)

        return math.log(probability)

    def score_sentence(self, sentence):
        """
        Calculate the log probability of a complete sentence.
        """

        words = ["<START>", "<START>"] + sentence + ["<END>"]

        score = 0.0

        for i in range(2, len(words)):
            score += self.log_trigram_probability(
                words[i - 2],
                words[i - 1],
                words[i]
            )

        return score

if __name__ == "__main__":

    from corpus import load_brown, split_brown

    sentences = load_brown()

    train_sentences, _ = split_brown(sentences)

    # Remove POS tags
    train_words = [
        [word for word, tag in sentence]
        for sentence in train_sentences
    ]

    model = TrigramLanguageModel()
    model.train(train_words)

    print("Vocabulary size:", len(model.vocabulary))
    print("Total words:", model.total_words)

    print("\nExample trigram probabilities:")

    print(
        "P(County | Fulton, The) =",
        model.trigram_probability(
            "The",
            "Fulton",
            "County"
        )
    )

    print(
        "P(Jury | County, Grand) =",
        model.trigram_probability(
            "County",
            "Grand",
            "Jury"
        )
    )

    print("\nSentence score:")

    example = [
        "The",
        "Fulton",
        "County",
        "Grand",
        "Jury"
    ]

    print(model.score_sentence(example))

if __name__ == "__main__":

    from corpus import load_spanish

    print("Loading Spanish-GSD...")

    train_sentences, _, _ = load_spanish()

    # Extract only the word forms.
    train_words = [
        [token["form"].lower() for token in sentence]
        for sentence in train_sentences
    ]

    print("Training Spanish trigram model...")

    model = TrigramLanguageModel()
    model.train(train_words)

    print("Vocabulary size:", len(model.vocabulary))
    print("Total words:", model.total_words)
