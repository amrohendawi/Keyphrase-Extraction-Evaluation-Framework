import json

import spacy
import pytextrank
from spacy import displacy
import re
import pandas as pd

from metrics.MyEvaluator import MyEvaluator
myEvaluator = MyEvaluator()

# load the corpus data from data folder as unicode format
with open('data/job_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# load the ground truth data from data folder
with open('data/job_keywords.json', 'r', encoding='utf-8') as f:
    ground_truth = json.load(f)


# clean up the text by removing multiple newlines for the sake of visualization and readability
def clean_text(text):
    text = re.sub(r'\n{2,}', '\n', text)
    return text


# clean up the text in the corpus data
for i in range(len(data)):
    data[i]['title'] = clean_text(data[i]['title'])
    data[i]['description'] = clean_text(data[i]['description'])

# load two spaCy german models and add PyTextRank to the spaCy pipeline
## small model
nlp_sm = spacy.load("de_core_news_sm")
nlp_sm.add_pipe("textrank")

## large model
nlp_lg = spacy.load("de_core_news_lg")
nlp_lg.add_pipe("textrank")


def extract_keyphrases(nlp, text):
    doc = nlp(text)
    keywords = [p.text for p in doc._.phrases]
    keywords = [keyword.lower() for keyword in keywords]
    return keywords


if __name__ == '__main__':
    # mrr = calculate_mrr(extract_keyphrases(nlp_sm, data[0]['description']), ground_truth_keyphrases)

    # create a dataframe to store the results as a table of the metrics
    # create a dataframe to store the results as a table of the metrics
    evaluation_results = pd.DataFrame(
        columns=['id', 'precision_sm', 'recall_sm', 'f1_sm', 'rouge_sm', 'overall_sm', 'precision_lg', 'recall_lg',
                 'f1_lg', 'rouge_lg', 'overall_lg', ])

    # iterate over the corpus data, extract keyphrases and calculate the metrics
    for i in range(len(data)):
        # make all keyphrases lowercase
        ground_truth_keyphrases = ground_truth[i]['description']
        ground_truth_keyphrases = [keyphrase.lower() for keyphrase in ground_truth_keyphrases]

        keyphrases_sm = extract_keyphrases(nlp_sm, data[i]['description'])
        keyphrases_lg = extract_keyphrases(nlp_lg, data[i]['description'])

        precision_sm, recall_sm, f1_sm = myEvaluator.calculate_all_metrics(prediction=keyphrases_sm,
                                                                           ground_truth=ground_truth_keyphrases)
        precision_lg, recall_lg, f1_lg = myEvaluator.calculate_all_metrics(prediction=keyphrases_lg,
                                                                           ground_truth=ground_truth_keyphrases)

        rouge_sm = myEvaluator.rouge_score(prediction=keyphrases_sm, ground_truth=ground_truth_keyphrases)
        rouge_lg = myEvaluator.rouge_score(prediction=keyphrases_lg, ground_truth=ground_truth_keyphrases)

        overall_sm = myEvaluator.overall_score(prediction=keyphrases_sm, ground_truth=ground_truth_keyphrases)
        overall_lg = myEvaluator.overall_score(prediction=keyphrases_lg, ground_truth=ground_truth_keyphrases)

        evaluation_results.loc[i] = [data[i]['id'], precision_sm, recall_sm, f1_sm, rouge_sm, overall_sm, precision_lg,
                                     recall_lg, f1_lg, rouge_lg, overall_lg]

    # print the results
    print(f"evaluation_results: {evaluation_results}")

    # calculate the average overall score for both models
    print(f"average overall score for small model: {evaluation_results['overall_sm'].mean()}")
    print(f"average overall score for large model: {evaluation_results['overall_lg'].mean()}")

    # plot the mean overall score for both models and the distribution of the overall scores
    evaluation_results[['overall_sm', 'overall_lg']].plot(kind='bar', title='Mean overall score for both models')
    evaluation_results[['overall_sm', 'overall_lg']].plot(kind='hist', title='Distribution of the overall scores')

    # plot the distribution of the overall scores for both models
    evaluation_results[['overall_sm']].plot(kind='hist', title='Distribution of the overall scores for small model')
    evaluation_results[['overall_lg']].plot(kind='hist', title='Distribution of the overall scores for large model')

    # plot a gauß distribution of the overall scores for both models
    evaluation_results[['overall_sm']].plot(kind='kde', title='Gauß distribution of the overall scores for small model')
    evaluation_results[['overall_lg']].plot(kind='kde', title='Gauß distribution of the overall scores for large model')

    ground_truth_keywords = ground_truth[0]['description']
    ground_truth_keywords = [keyword.lower() for keyword in ground_truth_keywords]

    keyphrases_sm = nlp_sm(data[0]['description'])
    keyphrases_lg = nlp_lg(data[0]['description'])

    displacy.serve(keyphrases_sm, style="ent")
