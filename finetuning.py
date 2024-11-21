# -*- coding: utf-8 -*-

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
from datasets import load_dataset, ClassLabel
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# from transformers.utils import logging
# logging.set_verbosity_error()
from sklearn.metrics import classification_report
# import numpy as np
import json
from src.config import my_cache_dir, my_output_dir
from src.utils import compute_metrics
from src.cross_validation import initialize_scores_dict, sum_cv_scores, mean_cv_scores
from src.custom_dataset import data_path, task_names, column_names, cv_number, hp

def main():

    def tokenize_data(example):
#        return tokenizer(example['text'], padding= 'max_length', truncation=True)
        return tokenizer(example['text'], padding= True, truncation=True)

    # Decode multiclass labels 
    def map_label2id(example):
        example['labels'] = ClassLabels.str2int(example[task])
        return example

    # Input method of finetunig
    method = sys.argv[1]
    if method not in ["baseline", "bestrun"]:
        print("{} is an unknown method. Please choose between baseline or bestrun")
        exit()


    # Input task name
    task = sys.argv[2]
    if task not in task_names: 
        print("{} is an unknown task. Please choose between {}".format(task, task_names))
        exit()

    # Input model name
    checkpoint = sys.argv[3]


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

    # Bestrun
    if method == "bestrun":
        if checkpoint in ["deepset/gbert-base", "dbmdz/bert-base-german-cased", "dbmdz/bert-base-german-uncased"]:
            hyperparameters = hp[method][task][checkpoint]
            print("Bestrun hyperparameters are {}\n".format(json.dumps(hyperparameters, indent=4)))
        else:
            print("Error: Bestrun hyperparameters for {} are not available. Please run hyperpameter search.\n".format(checkpoint))
            exit()
    else:
        hyperparameters = {'learning_rate': 2e-05,
                           'num_train_epochs': 10,
                           'seed': 123,
                           'per_device_train_batch_size': 16,
                           'weight_decay': 0.01,
                           "adam_epsilon": 1e-08,
                           "gradient_accumulation_steps": 1
                           }


    # save results to folder results
    file_name = "results/{}_{}_{}.txt".format(method, task, checkpoint.split("/")[-1])
    w = open(file_name, "w+", encoding="utf-8")
    w.write("{}\n\n".format(checkpoint))

    # for scores from cross-validation
    cv_results = initialize_scores_dict(labels)


    # start cross-validation
    for n in range(0,fold):
        print("\n****************************Cross-validation number {}****************************\n".format(n+1))
        # split dataset
        dataset_train = load_dataset("json", data_files="data/{}_cv{}.json".format(task.replace("sa_", ""), str(n)), field="train")
        dataset_valid = load_dataset("json", data_files="data/{}_cv{}.json".format(task.replace("sa_", ""), str(n)), field="test")
        dataset_train = dataset_train['train']
        dataset_valid = dataset_valid['train']
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=128, cache_dir=my_cache_dir)

        dataset_train = dataset_train.map(tokenize_data, batched=True)
        dataset_valid = dataset_valid.map(tokenize_data, batched=True)
        valid_labels = dataset_valid[task]
        remove_columns = column_names
        dataset_train = dataset_train.map(map_label2id, remove_columns=remove_columns)
        dataset_valid = dataset_valid.map(map_label2id, remove_columns=remove_columns)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # new

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_lab, cache_dir=my_cache_dir
        )

        training_args = TrainingArguments(
            learning_rate=hyperparameters["learning_rate"],
            output_dir=my_output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=hyperparameters["per_device_train_batch_size"],
            num_train_epochs=hyperparameters["num_train_epochs"],
            seed=hyperparameters["seed"],
            adam_epsilon=hyperparameters["adam_epsilon"],
            weight_decay=hyperparameters["weight_decay"],
            warmup_ratio=0.1,
            optim="adamw_torch",
            eval_strategy="epoch",
            save_strategy="epoch",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_train,
            eval_dataset=dataset_valid,
            processing_class=tokenizer,
            data_collator=data_collator, # new
            compute_metrics=compute_metrics,
#            data_seed=123,

        )

        trainer.train()

        # Classification report for validation set
        results = trainer.evaluate()
        predicted_results=trainer.predict(dataset_valid)
        predicted_labels = predicted_results.predictions.argmax(-1) # Get the highest probability prediction
        predicted_labels = predicted_labels.flatten().tolist()      # Flatten the predictions into a 1D list
        predicted_labels = [ClassLabels.int2str(l) for l in predicted_labels]  # Convert from integers back to strings for readability

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
                    string +=  '%*.*f' % (11,4, value2)
                else: string +=  '%*.*f' % (11,2, value2)
            output += '%12s%s\n' % (key, string)
        else:
            output += '\n%12s%*.*f' % ('accuracy', 33, 4, cv_results['accuracy'])
            output += '%*.*f\n' % (11,2, cv_results['macro avg']['support'])
    print(output)

    w.write(output)
    w.close()


    # Training arguments
#    print(training_args.to_json_string())
if __name__ == "__main__":
    main()