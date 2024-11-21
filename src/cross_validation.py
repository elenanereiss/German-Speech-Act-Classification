# -*- coding: utf-8 -*-


def initialize_scores_dict(labels):
    d = {} 
    for label in labels:
        d[label] = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
    d.update({'accuracy': [], 'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}, 'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []}})
    return d


# collect all scores from cross validation
def sum_cv_scores(cv_results, cv_number):
    for item in cv_number.keys():
        if item != 'accuracy':
            for score in cv_number[item].keys():
                cv_results[item][score].append(cv_number[item][score])
    cv_results['accuracy'].append(cv_number['accuracy'])
    return cv_results


# calculate mean of scores for cross validation
def mean_cv_scores(cv_results):
#    print(cv_results)
    for item in cv_results.keys():
        if item != 'accuracy':
            for score in cv_results[item].keys():
                cv_results[item][score] = sum(cv_results[item][score])/len(cv_results[item][score])
    cv_results['accuracy'] = sum(cv_results['accuracy'])/len(cv_results['accuracy'])
    return cv_results


