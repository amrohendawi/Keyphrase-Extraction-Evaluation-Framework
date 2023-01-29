class ClassicMetrics:
    def __init__(self):
        pass

    def calculate_all_metrics(self, prediction: list, ground_truth: list) -> (float, float, float):
        """
        Calculate the precision, recall and f1 score of the predicted keyphrases
        :param prediction: list of predicted keyphrases
        :param ground_truth: list of ground truth keyphrases
        :return: precision: float, recall: float, f1: float
        """
        TP, FP, FN = 0, 0, 0

        for keyphrase in prediction:
            if keyphrase in ground_truth:
                TP += 1
            else:
                FP += 1

        for keyphrase in ground_truth:
            if keyphrase not in prediction:
                FN += 1

        # Calculate precision and recall
        precision = 0.0 if TP == 0 else TP / (TP + FP)
        recall = 0.0 if TP == 0 else TP / (TP + FN)

        # Calculate F1 score
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

        # format the metrics to 4 floating points
        precision = float("{:.4f}".format(precision))
        recall = float("{:.4f}".format(recall))
        f1 = float("{:.4f}".format(f1))

        return precision, recall, f1
