from metrics.ClassicMetrics import ClassicMetrics
from metrics.Rouge import Rouge


class MyEvaluator(ClassicMetrics, Rouge):
    def __init__(self, n=1):
        super().__init__()
        Rouge.__init__(self, n)

    def overall_score(self, prediction: list, ground_truth: list, w_f1=0.5, w_rouge=0.5) -> float:
        """
        Calculate the overall score of the predicted keyphrases
        :param prediction: list of predicted keyphrases
        :param ground_truth: list of ground truth keyphrases
        :param w_f1: weight for F1 score. Default is 0.5
        :param w_rouge: weight for ROUGE score. Default is 0.5
        :return: overall score
        """
        precision, recall, f1 = self.calculate_all_metrics(prediction, ground_truth)
        rouge_n = self.rouge_score(prediction, ground_truth)

        overall_score = w_f1 * f1 + w_rouge * rouge_n

        return overall_score
