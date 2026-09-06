from collections import Counter, defaultdict
import math


class TrigramPOSTagger:
    def __init__(self):
        # Counts for P(word | tag)
        self.word_tag_counts = Counter()
        self.tag_counts = Counter()

        # Counts for P(tag3 | tag1, tag2)
        self.tag_bigram_counts = Counter()
        self.tag_trigram_counts = Counter()

        # All tags seen during training
        self.tags = set()

        # Vocabulary
        self.vocabulary = set()
        self.word_to_tags = defaultdict(set)

    def train(self, sentences):
        """
        Train the POS tagger.

        sentences:
            List of sentences.
            Each sentence is a list of (word, tag) tuples.
        """

        for sentence in sentences:

            # Count emissions
            for word, tag in sentence:
                word = word.lower()

                self.word_tag_counts[(word, tag)] += 1
                self.word_to_tags[word].add(tag)
                self.tag_counts[tag] += 1

                self.tags.add(tag)
                self.vocabulary.add(word)

            # Add two start tags.
            tags = ["<START>", "<START>"] + [
                tag for word, tag in sentence
            ]

            # Count tag bigrams and trigrams.
            for i in range(2, len(tags)):

                previous_two = (tags[i - 2], tags[i - 1])
                current_tag = tags[i]

                self.tag_bigram_counts[previous_two] += 1

                self.tag_trigram_counts[
                    (tags[i - 2], tags[i - 1], current_tag)
                ] += 1

    def emission_probability(self, word, tag):
        """
        Calculate P(word | tag).
        """

        word = word.lower()

        numerator = self.word_tag_counts[(word, tag)] + 1

        denominator = (
            self.tag_counts[tag]
            + len(self.vocabulary)
        )

        return numerator / denominator

    def log_emission_probability(self, word, tag):
        return math.log(
            self.emission_probability(word, tag)
        )

    def transition_probability(self, tag1, tag2, tag3):
        """
        Calculate P(tag3 | tag1, tag2).
        """

        numerator = (
            self.tag_trigram_counts[
                (tag1, tag2, tag3)
            ] + 1
        )

        denominator = (
            self.tag_bigram_counts[(tag1, tag2)]
            + len(self.tags)
        )

        return numerator / denominator

    def log_transition_probability(
        self,
        tag1,
        tag2,
        tag3
    ):
        return math.log(
            self.transition_probability(
                tag1,
                tag2,
                tag3
            )
        )

    def possible_tags(self, word):
        """
        Return tags observed with this word.

        For an unknown word, return every tag.
        """

        word = word.lower()

        tags = self.word_to_tags.get(word)

        if not tags:
            return self.tags

        return tags

    def tag(self, words):
        """
        Tag a sequence of words using trigram Viterbi decoding.

        Returns:
            List of (word, tag) tuples.
        """

        if not words:
            return []

        # DP state:
        #
        # (previous_previous_tag, previous_tag)
        #
        # -> (score, previous_state, chosen_tag)

        dp = {
            ("<START>", "<START>"): (
                0.0,
                None,
                None
            )
        }

        backpointers = []

        for word in words:

            new_dp = {}

            for (
                prev_prev_tag,
                prev_tag
            ), (
                score,
                _,
                _
            ) in dp.items():

                candidate_tags = self.possible_tags(word)

                for tag in candidate_tags:

                    transition_score = (
                        self.log_transition_probability(
                            prev_prev_tag,
                            prev_tag,
                            tag
                        )
                    )

                    emission_score = (
                        self.log_emission_probability(
                            word,
                            tag
                        )
                    )

                    new_score = (
                        score
                        + transition_score
                        + emission_score
                    )

                    new_state = (
                        prev_tag,
                        tag
                    )

                    if (
                        new_state not in new_dp
                        or new_score > new_dp[new_state][0]
                    ):
                        new_dp[new_state] = (
                            new_score,
                            (
                                prev_prev_tag,
                                prev_tag
                            ),
                            tag
                        )

            dp = new_dp
            backpointers.append(dp)

        # Find the best final state.
        best_state = max(
            dp,
            key=lambda state: dp[state][0]
        )

        # Reconstruct tags backwards.
        predicted_tags = []

        for i in range(
            len(words) - 1,
            -1,
            -1
        ):

            state_data = backpointers[i][best_state]

            _, previous_state, tag = state_data

            predicted_tags.append(tag)

            best_state = previous_state

        predicted_tags.reverse()

        return list(zip(words, predicted_tags))

class MorphologyAwarePOSTagger(TrigramPOSTagger):
    """
    POS tagger whose labels include morphological information.

    Example:
        NOUN + Gender=Fem + Number=Sing
        -> NOUN-Fem-Sg

        ADJ + Gender=Masc + Number=Plur
        -> ADJ-Masc-Pl
    """

    @staticmethod
    def make_morph_tag(upos, feats):
        """
        Construct a morphology-aware tag.

        Gender:
            Masc / Fem / Neut

        Number:
            Sing / Plur

        If gender or number is unavailable, it is simply omitted.
        """

        tag = upos

        gender = feats.get("Gender")
        number = feats.get("Number")

        if gender in {"Masc", "Fem", "Neut"}:
            gender_short = {
                "Masc": "Masc",
                "Fem": "Fem",
                "Neut": "Neut"
            }[gender]

            tag += f"-{gender_short}"

        if number in {"Sing", "Plur"}:
            number_short = {
                "Sing": "Sg",
                "Plur": "Pl"
            }[number]

            tag += f"-{number_short}"

        return tag

    def train_spanish(self, sentences):
        """
        Train using Spanish-GSD sentences.

        Each sentence contains dictionaries with:
            form
            lemma
            upos
            feats
        """

        converted_sentences = []

        for sentence in sentences:

            converted_sentence = []

            for token in sentence:

                word = token["form"]
                upos = token["upos"]
                feats = token["feats"]

                morph_tag = self.make_morph_tag(
                    upos,
                    feats
                )

                converted_sentence.append(
                    (word, morph_tag)
                )

            converted_sentences.append(
                converted_sentence
            )

        # Use the existing trigram POS training machinery.
        self.train(converted_sentences)

        return converted_sentences

if __name__ == "__main__":

    from corpus import load_brown, split_brown

    print("Loading Brown corpus...")

    sentences = load_brown()

    train_sentences, _ = split_brown(sentences)

    print("Training POS tagger...")

    tagger = TrigramPOSTagger()
    tagger.train(train_sentences)

    test_sentence = [
        "the",
        "quick",
        "brown",
        "fox"
    ]

    result = tagger.tag(test_sentence)

    print("\nInput:")
    print(test_sentence)

    print("\nPOS tags:")

    for word, tag in result:
        print(f"{word} -> {tag}")

if __name__ == "__main__":

    from corpus import load_spanish

    print("Loading Spanish-GSD...")

    train_sentences, dev_sentences, test_sentences = load_spanish()

    print("Training morphology-aware POS tagger...")

    tagger = MorphologyAwarePOSTagger()

    converted_train = tagger.train_spanish(
        train_sentences
    )

    print("\nExample morphology-aware tags:")

    for word, tag in converted_train[0]:
        print(f"{word} -> {tag}")

    test_words = [
        "la",
        "casa",
        "roja",
        "es",
        "grande"
        ]

    result = tagger.tag(test_words)

    print("\nTest sentence:")
    print(test_words)

    print("\nPredicted tags:")

    for word, tag in result:
        print(f"{word} -> {tag}")
