import nltk

nltk.download('punkt')
from nltk.util import ngrams


class Rouge:

    def __init__(self, n=1):
        self.n = n

    def rouge_score(self, prediction: list, ground_truth: list) -> float:
        """
          This method calculates the ROUGE-N score for a given set of predicted and actual keyphrases while reflecting partial matches.
            The ROUGE-N score is calculated as follows:
            1. For each predicted keyphrase, generate all n-grams.
            2. For each actual keyphrase, generate all n-grams.
            3. Calculate the intersection of the predicted and actual n-grams.
            4. Calculate the ROUGE-N score as the sum of the intersection of predicted and actual n-grams divided by the total number of predicted n-grams.
        """
        rouge_score = 0.0
        total_predicted_keyphrases = 0
        for pk in prediction:
            pk_ngrams = set(ngrams(nltk.word_tokenize(pk), self.n))
            total_predicted_keyphrases += len(pk_ngrams)
            for ak in ground_truth:
                ak_ngrams = set(ngrams(nltk.word_tokenize(ak), self.n))
                rouge_score += len(pk_ngrams.intersection(ak_ngrams))
        rouge_score = 0.0 if rouge_score == 0 else rouge_score / total_predicted_keyphrases
        return rouge_score
