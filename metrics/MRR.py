class MRR:
    def __init__(self):
        pass

    def calculate_mrr(self, predicted_keyphrases: list, ground_truth: list) -> float:
        """
        Calculate the mean reciprocal rank (MRR) of the predicted keyphrases
        :param predicted_keyphrases: a list of predicted keyphrases sorted by their rank (highest rank first)
        :param ground_truth: a list of ground truth keyphrases
        :return:
        """
        mrr = 0
        for i, keyphrase in enumerate(predicted_keyphrases):
            if keyphrase in ground_truth:
                mrr += 1 / (i + 1)
        mrr = mrr / len(predicted_keyphrases)
        return mrr
