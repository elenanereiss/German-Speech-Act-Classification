# -*- coding: utf-8 -*-

import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
from datasets import load_dataset, ClassLabel, Sequence
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from sklearn.metrics import classification_report
import numpy as np
import json
from src.config import my_cache_dir, my_output_dir
from src.utils import compute_metrics
# from src.cross_validation import initialize_scores_dict, sum_cv_scores, mean_cv_scores
from src.custom_dataset import data_path, task_names, column_names, cv_number
import ray
ray.init(ignore_reinit_error=True, num_cpus=4)
print("success")


def main():

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_lab,
                                                                  cache_dir=my_cache_dir)
        # , return_dict=True)

    def my_hp_space(trial):
        #    gc.collect()
        #    torch.cuda.empty_cache()
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 4e-4, log=True),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 15),
            "seed": trial.suggest_int("seed", 1, 40),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
            "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
            "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4, 8]),
        }

    def my_objective(metrics):
        # Your elaborate computation here
        #    return metrics['eval_loss']
        return metrics['eval_f1']

    def tokenize_data(example):
        return tokenizer(example['text'], padding=True, truncation=True)

    # Decode multiclass labels
    def map_label2id(example):
        example['labels'] = ClassLabels.str2int(example[task])
        return example

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

    dataset_train = load_dataset("json", data_files="data/{}_cv0.json".format(task.replace("sa_", "")), field="train")
    dataset_valid = load_dataset("json", data_files="data/{}_cv0.json".format(task.replace("sa_", "")), field="test")
    dataset_train = dataset_train['train']
    dataset_valid = dataset_valid['train']

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, model_max_length=128, cache_dir=my_cache_dir)

    dataset_train = dataset_train.map(tokenize_data, batched=True)
    dataset_valid = dataset_valid.map(tokenize_data, batched=True)
#    valid_labels = dataset_valid[task]
    remove_columns = column_names
    dataset_train = dataset_train.map(map_label2id, remove_columns=remove_columns)
    dataset_valid = dataset_valid.map(map_label2id, remove_columns=remove_columns)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # new

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_lab, cache_dir=my_cache_dir
    )


    training_args = TrainingArguments(
        eval_strategy="steps",
        eval_steps=500,
        disable_tqdm=True,
        output_dir=my_output_dir,
    )

    trainer = Trainer(
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        processing_class=tokenizer,
#        tokinizer=tokenizer,
        data_collator=data_collator,  # new
        model_init=model_init,
        compute_metrics=compute_metrics,
    )

    result = trainer.hyperparameter_search(direction="maximize",n_trials=30,hp_space=my_hp_space, compute_objective=my_objective)
    print(result)


if __name__ == "__main__":
    main()


