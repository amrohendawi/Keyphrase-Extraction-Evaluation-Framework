# Evaluation framework for keyphrase extraction methods

This repository contains the evaluation framework for keyphrase extraction methods.
For the demonstration we use [spacy](https://spacy.io/) and [pytextrank](https://github.com/DerwenAI/pytextrank) as a
demo keyphrase extraction method.

The goal is to implement an evaluation-framework for different approaches of automatic key-phrase
extraction. With this scope selecting a meaningful evaluation measure is of peculiar interest. In order to test and/or
train any machine-learning model there is a set of 100 job ads available in [here](data/job_keywords.json).

## Essentials

+ Loading and potentially pre-processing data
+ Tagging/Key-phrase-extraction
+ Evaluation of the quality of a model
+ Simple evaluation framework for different approaches/models


## Evaluation framework

The ad-hoc evaluation framework uses `job_keywords.json` as a ground truth and `job_data.json` as a corpus.

The evaluation framework contains implementations of the following metrics:

1. Precision
2. Recall
3. F1
4. ROUGE-L
5. MRR

The final designed evaluation metric is the weighted average of `F1` & `ROUGE` scores with weights `0.5` & `0.5` respectively as in the following formula:

`final_score = 0.5 * F1 + 0.5 * ROUGE`

F1 is a result of both precision and recall. That's why we use it as a part of the final score.

## Evaluation metrics

### Standard metrics

1. **Precision:** This metric measures the proportion of predicted keyphrases that are correct, i.e., that appear in the
   reference set of keyphrases. A high precision score indicates that the model is able to identify most of the
   keyphrases correctly, but it may miss some keyphrases if it is too conservative.
2. **Recall:** This metric measures the proportion of reference keyphrases that are correctly identified by the model,
   i.e.,
   that are included in the predicted set of keyphrases. A high recall score indicates that the model is able to
   identify most of the keyphrases, but it may also include some false positive keyphrases if it is too liberal.
3. **F1 score:** This metric is the harmonic mean of precision and recall, and it balances the two metrics by taking
   into
   account both the false positives and false negatives. A high F1 score indicates that the model is able to identify a
   good balance of the reference keyphrases while minimizing false positives.

### Extended metrics

1. **Mean reciprocal rank (MRR):** This metric measures the average of the reciprocal ranks of the first correct
   keyphrase
   in the predicted list for each document. A high MRR score indicates that the model is able to identify the correct
   keyphrases close to the top of the list.
2. **Coverage:** This metric measures the proportion of the reference keyphrases that are included in the predicted
   keyphrase set. A high coverage score indicates that the model is able to identify a large proportion of the
   keyphrases, but it may also include a large number of false positive keyphrases.
3. **ROUGE:** This metric measures the similarity between the predicted keyphrases and the reference keyphrases. A high
   ROUGE score indicates that the model is able to identify a good balance of the reference keyphrases while
   minimizing false positives.

In general, the best metric to evaluate keyphrase extraction models will depend on the specific goals and constraints of
the task at hand. For example, if the goal is to identify as many of the keyphrases as possible, a metric like recall or
coverage may be more relevant.

On the other hand, if the goal is to minimize false positives, a metric like precision or
F1 score may be more relevant. It is often useful to use multiple metrics to get a more complete picture of the model's
performance.

### Critical analysis

The standard metrics like accuracy, precision, recall, and F1 score do not reflect partial matches, but rather only
consider the perfect match between an extracted segment and the correct prediction for that tag.

For example, if the correct prediction for a tag is `New York City`, and the extracted segment is `New York`, the
standard
metrics will not consider this a correct prediction. However, in many cases, the extracted segment is not exactly the
same
as the correct prediction, but it is still a correct prediction.

## Extended Analysis of the Evaluation Framework (January 2023)

In conclusion, a combination of precision, recall, F1-score and ROUGE as the evaluation metrics.
As the goal of the task is to extract key-phrases from job advertisements, precision and recall would be used to
evaluate the ability of the model to extract the key-phrases that are in the job advertisement but also the ones that
are not, F1-score will be helpful to balance those two measurements. Finally, ROUGE would be used to evaluate the
overall quality of the extracted key-phrases, by measuring the overlap between the predicted and actual key-phrases.

### ROUGE with partial matches

Let's say we have predicted_keyphrases = `["recruiting online", "job ads"]` and actual_keyphrases
= `["job ads", "recruiting process"]`

First, we need to tokenize the keyphrases and then generate 1-grams and 2-grams for both predicted and actual
keyphrases.

1. For n = 1:

   N_predicted = {("recruiting",), ("online",), ("job",), ("ads",)}
   
   N_actual = {("job",), ("ads",), ("recruiting",), ("process",)}

   `ROUGE-1 = |Np ∩ Na| / |Na| = 3/4 = 0.75`


2. For n = 2:

   N_predicted = {("recruiting", "online"), ("job", "ads")}
   
   N_actual = {("job", "ads"), ("recruiting", "process")}
   
   `ROUGE-2 = |Np ∩ Na| / |Na| = 1/2 = 0.5`

### Paper references

- [PatternRank: Leveraging Pretrained Language Models and Part of Speech for Unsupervised Keyphrase Extraction](https://paperswithcode.com/paper/patternrank-leveraging-pretrained-language)
- [Key2Vec: Automatic Ranked Keyphrase Extraction from Scientific Articles using Phrase Embeddings](https://arxiv.org/abs/2003.04628)
- [Large-Scale Evaluation of Keyphrase Extraction Models](https://arxiv.org/pdf/2003.04628.pdf)

## Installation

First create an environment with python 3.9 and activate it.
You can use `environment.yml` to create **conda** environment or `requirements.txt` to create **pip** environment.

### Conda

```bash
conda env create -f environment.yml
conda activate eval_framework_kpe
```

### Pip

```bash
python -m venv eval_framework_kpe
source eval_framework_kpe/bin/activate
pip install -r requirements.txt
```

## Testing the framework metrics

The unit tests are located in metrics_test.py. To run the tests, use the following command:

```bash
python -m unittest metrics_test.py
```

## Running the demo

The demo is a jupyter notebook located in demo.ipynb. To run the demo, use the following command:

```bash
jupyter notebook demo.ipynb
```

Make sure you have jupyter notebook installed in your environment.