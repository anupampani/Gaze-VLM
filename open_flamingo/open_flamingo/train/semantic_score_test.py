from evaluation_utils import sentence_transformers_similarity


def main():
    reference_texts = ["This trial.", "Here is another sample."]
    generated_texts = ["This is a trial.", "Here is a different sample."]

    predictions = [
"\xa0 \xa0 \xa0  . The hand is on a floor, other seated laid. ",
                "The to card is placed onto catches to a on a cards in the table.",
                " AThe',  \xa0 \xa0 \xa0  .The hand shows blurred-blurred, with a quick movement. likely the cards throwing a from",
                "The image's holding a deck of hands- in with concluding their game.",
                "The hand on hand is in, a card of cards cards cards, a cards in on the table.",
                "A person is sitting a set of playing cards in their hand,"

]



    paraphrases = [
        "A card settles on the table, merging with the remaining cards.",
        "Another card is tossed and lands alongside the earlier ones on the table.",
        "The picture is blurred from rapid motion, likely from shuffling or relocating cards.",
        "The individual clutches two face cards, potentially contemplating their next move.",
        "Visible is a hand displaying an assortment of various playing cards, with more cards laid out on the table.",
        "An individual grips a spread of playing cards in their hands."

    ]


    references = [
        "The card lands on the table joining the other cards.",
        "Yet another card is thrown and coming to rest with previous ones on the table.",
        "The image is motion-blurred, capturing a quick movement, possibly dealing or moving cards.",
        "The person is holding a pair of face cards, possibly considering a play.",
        "The person's hand is visible with a spread of mixed playing cards and additional cards placed on the table.",
        "A person is holding a fan of playing cards in their hands."
        # Add the rest of your ground truth descriptions
    ]
    scores = sentence_transformers_similarity(references, paraphrases)

    print("Similarity Scores:", scores)

if __name__ == "__main__":
    print("yoooo")
    main()


# Similarity Scores: [0.3982318937778473, 0.6032022833824158, 0.6862139105796814, 0.5610337257385254, 0.7957852482795715, 0.7443492412567139]

# Similarity Scores: [0.2544151842594147, 0.7106633186340332, 0.7354191541671753, 0.4934060573577881, 0.7030794024467468, 0.8028706908226013]