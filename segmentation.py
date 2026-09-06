import math


class ViterbiSegmenter:
    def __init__(self, language_model, max_word_length=20):
        self.lm = language_model
        self.max_word_length = max_word_length

    def segment(self, text):
        """
        Segment a string using trigram probabilities and
        dynamic programming.

        Returns:
            List of words representing the best segmentation.
        """

        text = text.lower()

        # DP state:
        #
        # (position, previous_word, previous_previous_word)
        #
        # -> (best_score, previous_state, chosen_word)

        states = {
            (0, "<START>", "<START>"): (0.0, None, None)
        }

        # Process positions from left to right.
        for position in range(len(text)):

            current_states = [
                state for state in states
                if state[0] == position
            ]

            for state in current_states:

                _, prev_prev_word, prev_word = state
                score, _, _ = states[state]

                # Try every possible word beginning at this position.
                max_length = min(
                    self.max_word_length,
                    len(text) - position
                )

                for length in range(1, max_length + 1):

                    candidate = text[position:position + length]

                    # Candidate must exist in vocabulary.
                    if candidate not in self.lm.vocabulary:
                        continue

                    new_position = position + length

                    word_score = self.lm.log_trigram_probability(
                        prev_prev_word,
                        prev_word,
                        candidate
                    )

                    new_score = score + word_score

                    new_state = (
                        new_position,
                        prev_word,
                        candidate
                    )

                    # Keep only the best path to this state.
                    if (
                        new_state not in states
                        or new_score > states[new_state][0]
                    ):
                        states[new_state] = (
                            new_score,
                            state,
                            candidate
                        )

        # Find the best completed state.
        final_states = [
            state for state in states
            if state[0] == len(text)
        ]

        if not final_states:
            return []

        best_state = max(
            final_states,
            key=lambda state: (
                states[state][0]
                + self.lm.log_trigram_probability(
                    state[1],
                    state[2],
                    "<END>"
                )
            )
        )

        # Reconstruct the segmentation.
        words = []

        current_state = best_state

        while current_state is not None:

            score, previous_state, chosen_word = states[current_state]

            if chosen_word is not None:
                words.append(chosen_word)

            current_state = previous_state

        words.reverse()

        return words

if __name__ == "__main__":

    from corpus import load_brown, split_brown
    from language_model import TrigramLanguageModel

    print("Loading Brown corpus...")

    sentences = load_brown()
    train_sentences, _ = split_brown(sentences)

    # Extract words only.
    train_words = [
        [word for word, tag in sentence]
        for sentence in train_sentences
    ]

    print("Training trigram model...")

    lm = TrigramLanguageModel()
    lm.train(train_words)

    segmenter = ViterbiSegmenter(
        language_model=lm,
        max_word_length=20
    )

    test_string = "thequickbrownfox"

    print("\nInput:")
    print(test_string)

    result = segmenter.segment(test_string)

    print("\nSegmentation:")
    print(result)

if __name__ == "__main__":

    from corpus import load_spanish
    from language_model import TrigramLanguageModel

    print("Loading Spanish-GSD...")

    train_sentences, _, _ = load_spanish()

    # Extract word forms.
    train_words = [
        [token["form"].lower() for token in sentence]
        for sentence in train_sentences
    ]

    print("Training Spanish trigram model...")

    lm = TrigramLanguageModel()
    lm.train(train_words)

    segmenter = ViterbiSegmenter(
        language_model=lm,
        max_word_length=20
    )

    test_string = "lacasarojaesgrande"

    print("\nInput:")
    print(test_string)

    result = segmenter.segment(test_string)

    print("\nSegmentation:")
    print(result)
