# -*- coding: utf-8 -*-

import evaluate
from datasets import ClassLabel
import numpy as np


# Metrics for evaluation
def compute_metrics(eval_pred):
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("f1")
    metric4 = evaluate.load("accuracy")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    precision = metric1.compute(predictions=predictions, references=labels, average="macro", zero_division=0)["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="macro", zero_division=0)["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="macro")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


