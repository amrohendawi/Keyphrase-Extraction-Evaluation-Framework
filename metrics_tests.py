import unittest
from metrics.Rouge import Rouge
from metrics.ClassicMetrics import ClassicMetrics
from metrics.MyEvaluator import MyEvaluator


class TestRougeEvaluator(unittest.TestCase):
    def setUp(self):
        self.predicted_keyphrases = ["recruiting online", "job ads"]
        self.actual_keyphrases = ["job ads", "recruiting process", "hiring process", "marketing"]
        self.rouge_evaluator = Rouge()

    def test_rouge_score(self):
        n_res_dict = {1: 0.75, 2: 0.5}
        for n, res in n_res_dict.items():
            self.rouge_evaluator.n = n
            rouge_score = self.rouge_evaluator.rouge_score(self.predicted_keyphrases, self.actual_keyphrases)
            print(f'ROUGE-{n} Score:', rouge_score)
            self.assertIsInstance(rouge_score, float)
            self.assertGreaterEqual(rouge_score, 0)
            self.assertLessEqual(rouge_score, 1)
            self.assertAlmostEqual(rouge_score, res, places=2)


class TestMetricsCalculator(unittest.TestCase):
    def setUp(self):
        self.predicted_keyphrases = ["recruiting", "online", "job ads"]
        self.ground_truth_keyphrases = ["job ads", "recruiting", "hiring process", "marketing"]
        self.metrics_calculator = ClassicMetrics()

    def test_all_metrics(self):
        precision, recall, f1 = self.metrics_calculator.calculate_all_metrics(self.predicted_keyphrases,
                                                                              self.ground_truth_keyphrases)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        self.assertIsInstance(float(precision), float)
        self.assertIsInstance(float(recall), float)
        self.assertIsInstance(float(f1), float)
        self.assertGreaterEqual(float(precision), 0)
        self.assertGreaterEqual(float(recall), 0)
        self.assertGreaterEqual(float(f1), 0)
        self.assertLessEqual(float(precision), 1)
        self.assertLessEqual(float(recall), 1)
        self.assertLessEqual(float(f1), 1)
        self.assertEqual(float(precision), 0.6667)
        self.assertEqual(float(recall), 0.5)
        self.assertEqual(float(f1), 0.5714)


class TestKeyphraseEvaluationFramework(unittest.TestCase):
    def setUp(self):
        self.evaluator = MyEvaluator()
        self.predicted_keyphrases = ["recruiting online", "job ads"]
        self.ground_truth_keyphrases = ["job ads", "recruiting process"]

    def test_all_metrics(self):
        precision, recall, f1 = self.evaluator.calculate_all_metrics(self.predicted_keyphrases, self.ground_truth_keyphrases)
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5)

    def test_rouge_score(self):
        rouge_score = self.evaluator.rouge_score(self.predicted_keyphrases, self.ground_truth_keyphrases)
        self.assertEqual(rouge_score, 0.75)

    def test_overall_score(self):
        overall_score = self.evaluator.overall_score(self.predicted_keyphrases, self.ground_truth_keyphrases)
        self.assertEqual(overall_score, 0.625)


if __name__ == '__main__':
    unittest.main()
