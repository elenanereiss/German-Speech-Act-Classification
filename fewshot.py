# -*- coding: utf-8 -*-

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
from datasets import load_dataset, ClassLabel, Sequence
from fastfit import FastFit, FastFitTrainer #, sample_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, pipeline
from transformers.utils import logging
logging.set_verbosity_error()
import evaluate
from sklearn.metrics import classification_report
import numpy as np
import json
from src.config import my_cache_dir, my_output_dir, my_models_path
from src.utils import compute_metrics
from src.cross_validation import initialize_scores_dict, sum_cv_scores, mean_cv_scores
from src.custom_dataset import data_path, task_names, column_names, cv_number, hp

def main():
    # Input method
    method = "fastfit"


    # Input task name
    task = sys.argv[1]
    if task not in task_names: 
        print("{} is an unknown task. Please choose between {}".format(task, task_names))
        exit()


    # Input model name
    checkpoint = sys.argv[2]

    # collect labels from dataset
    dataset = load_dataset("json", data_files=data_path["full"], field=data_path["field"])
    labels = sorted(set(label for label in dataset["train"][task]))
    
    num_lab = len(labels)

    # Cast to ClassLabel
    ClassLabels = ClassLabel(num_classes=len(labels), names=labels)

    scores_dict = {}

    print("\n\n\n****************************Training of {} {}****************************\n".format(method, checkpoint))

    # cross-validation
    fold = cv_number

    # save results to folder results
    file_name = "results/{}_{}_{}.txt".format(method, task, checkpoint.split("/")[-1])
    w = open(file_name, "w+", encoding="utf-8")
    w.write("{}\n\n".format(checkpoint))

    # for scores from cross-validation
    cv_results = initialize_scores_dict(labels)

    # start cross-validation
    for n in range(0,fold):
        print("\n****************************Cross-validation number {}****************************\n".format(n+1))

        # Load dataset
        dataset = load_dataset("json", data_files="data/{}_cv{}.json".format(task.replace("sa_", ""), str(n)), field="train")
        dataset_valid = load_dataset("json", data_files="data/{}_cv{}.json".format(task.replace("sa_", ""), str(n)), field="test")

    #   Few shot with 8 labels per class
    #    dataset["train"] = sample_dataset(dataset["train"], label_column="sa_coarse", num_samples_per_label=8)
        dataset["validation"] = dataset_valid['train']
        dataset["test"] = dataset_valid['train']

        # Gold labels
        valid_labels = dataset_valid['train'][task]
#        metrics = {"precision", "recall", "f1", "accuracy"}
        # start training for each model
        trainer = FastFitTrainer(
            model_name_or_path=checkpoint,
            label_column_name=task,
            text_column_name="text",
            num_train_epochs=40,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            max_text_length=128,
            dataloader_drop_last=False,
            num_repeats=4,
#            optim="adafactor",
            optim="adamw_torch",
            weight_decay=0.01, #new
            warmup_ratio=0.1,
            clf_loss_factor=0.1,
            fp16=True,
            dataset=dataset,
#            metric_name="f1",
        )


        model = trainer.train()
        trainer.export_model()
        model.save_pretrained(my_models_path)
        results = trainer.evaluate()    

        print(results)
    
        # Step 1: Load a pre-trained model from disk
        model = FastFit.from_pretrained(my_models_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=my_cache_dir)

        # Resolve the issue with TypeError: forward() got an unexpected keyword argument 'token_type_ids'
        if "token_type_ids" in tokenizer.model_input_names: tokenizer.model_input_names.remove("token_type_ids")
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device="cuda")

        # Step 2: Run predictions to calculate class level metrics
        predicted_labels = []
        for row in dataset["validation"]:
#            print(classifier(row["text"]))
            predicted_labels.append(classifier(row["text"])[0]["label"])


        cv_results = sum_cv_scores(cv_results, classification_report(valid_labels,predicted_labels,zero_division=0, digits=4, output_dict=True))
        
        results=classification_report(valid_labels,predicted_labels,zero_division=0, digits=4)
        print(results)

        # Write results to file
        w.write("**************************************************Cross-validation number {}**************************************************\n\n".format(n+1))
        w.write(results)
        w.write("\n")

    cv_results = mean_cv_scores(cv_results)

    output = "************************************************Results of 5 fold cross-validation************************************************\n\n"
    output += '%12s%11s%11s%11s%11s\n\n' % ('', 'precision', 'recall', 'f1-score', 'support')
    for key, value in cv_results.items():
        if key != 'accuracy':
            string= ""
            for key2, value2 in cv_results[key].items():
                if key2 != 'support':
                    string += * '%*.f' % (11,4, value2)
                else: string +=  '%*.*f' % (11,2, value2)
            output += '%12s%s\n' % (key, string)
        else:
            output += '\n%12s%*.*f' % ('accuracy', 33, 4, cv_results['accuracy'])
            output += '%*.*f\n' % (11,2, cv_results['macro avg']['support'])
    print(output)

    w.write(output)
    w.close()


if __name__ == "__main__":
   main()
    

