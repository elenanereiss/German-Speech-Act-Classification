# Automated Speech Act Classification in Offensive German Language Tweets



## Experiment setup

- **Dataset**: German Speech Acts Dataset
- **Granularity**: coarse- and modified fine-grained speech acts. 
- **Experiment methods**: 
 - baseline with [Huggingface](https://huggingface.co/) transformers;
 - hyperparameter search with Huggingface transformers and [Ray Tune](https://docs.ray.io/en/latest/tune/index.html);
 - few-shot classification with Huggingface transformers and [Fastfit](https://github.com/IBM/fastfit);
 - fine-tuning with [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini).
- **Selected models** for baseline, hyperparameter search and few-shot learning are:
 - [deepset/gbert-base](https://huggingface.co/deepset/gbert-base)
 - [dbmdz/bert-base-german-cased](https://huggingface.co/dbmdz/bert-base-german-cased)
 - [dbmdz/bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased)
- **Evaluation method**: 5-fold cross-validation, divided in 80% train split with 1567 sentences and 20% validation split with 392 sentences. For each granularity, we created individual splits.
- **Evaluation metric**: precision, recall, f1-macro. Results of each model, each granularity and each method are in the folder [`/results`](/results).




## Dataset

German Speech Acts Dataset can be found in [anon_link](XXX). For experiments, we used the latest version v1.1. For 5-fold cross-validation, we split dataset where sentences were shuffled and stratified in order to preserve the percentage of samples for each class in each fold and split. For coarse- and fine-grained labels, we have created individual splits (s. tables below). 


### Mean number of coarse-grained labels

|            |   train |   val  |   total |
|:-----------|--------:|-------:|--------:|
| Assertive  |     546 |    137 |     683 |
| Expressive |     320 |     80 |     400 |
| Commissive |      16 |      4 |      20 |
| Directive  |     506 |    127 |     633 |
| Unsure     |     120 |     30 |     150 |
| Other      |      59 |     14 |      73 |
| total      |    1567 |    392 |    1959 |

### Mean number of fine-grained labels 

**Please note:** Due to sparse occurrences of some fine-grained classes, only classes which occur more than ten times were included. *Disagree*, *apologize*, *thank*, *greet* were united in a class *excluded*. As for the coarse-grained class *commissive*, we have decided not to divide into fine-grained classes. Thus, the number of fine-grained speech acts was reduced from 23 to a 17. 

|              |   train |   val  |   total |
|:-------------|--------:|-------:|--------:|
| Assert       |     472 |    118 |     590 |
| Sustain      |      10 |      3 |      13 |
| Guess        |      22 |      5 |      27 |
| Predict      |      27 |      7 |      34 |
| Agree        |      10 |      3 |      13 |
| Rejoice      |      14 |      3 |      17 |
| Complain     |     206 |     51 |     257 |
| Wish         |       9 |      2 |      11 |
| Expressemoji |      85 |     21 |     106 |
| Commissive   |      16 |      4 |      20 |
| Request      |     130 |     33 |     163 |
| Require      |      62 |     16 |      78 |
| Suggest      |      13 |      3 |      16 |
| Address      |     300 |     75 |     375 |
| Unsure       |     120 |     30 |     150 |
| Other        |      58 |     15 |      73 |
| Excluded     |      13 |      3 |      16 |
| total        |    1567 |    392 |    1959 |


## Evaluation
### Baseline


For our baselines, we selected default hyperparameters: 

```json
        hyperparameters = {'learning_rate': 2e-05,
                           'num_train_epochs': 10,
                           'seed': 123,
                           'per_device_train_batch_size': 16,
                           'weight_decay': 0.01,
                           "adam_epsilon": 1e-08,
                           "gradient_accumulation_steps": 1
                           }
```
<!---
#### Coarse-grained labels

|           |   deepset/gbert-base |   dbmdz/bert-base-german-cased |   dbmdz/bert-base-german-uncased |
|:----------|---------------------:|-------------------------------:|---------------------------------:|
| precision |              68.8114 |                        67.0367 |                          70.4424 |
| recall    |              65.6223 |                        64.3296 |                          66.2102 |
| f1-score  |              66.5067 |                        65.05   |                          67.764  |

#### Fine-grained labels

|           |   deepset/gbert-base |   dbmdz/bert-base-german-cased |   dbmdz/bert-base-german-uncased |
|:----------|---------------------:|-------------------------------:|---------------------------------:|
| precision |              55.8016 |                        57.8166 |                          55.9067 |
| recall    |              48.3927 |                        50.7018 |                          49.8822 |
| f1-score  |              50.1383 |                        52.5486 |                          51.8392 |

-->   


### Hyperparameter Search

We performed a hyperparameter search on the first train and validation split used a Python library [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). The goal was to maximize F1-macro of baseline models during 30 trials. After finding the best hyperparameters, we trained and evaluated a model on 5-folds. Hyperparameter space was defined as follow:
```json
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1,15),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-12, 1e-1, log=True),
        "adam_epsilon": trial.suggest_float("adam_epsilon", 1e-10, 1e-6, log=True),
        "gradient_accumulation_steps": trial.suggest_categorical("gradient_accumulation_steps", [1,2,4,8,16]),
```


<!---
#### Coarse-grained labels

|           |   deepset/gbert-base |   dbmdz/bert-base-german-cased |   dbmdz/bert-base-german-uncased |
|:----------|---------------------:|-------------------------------:|---------------------------------:|
| precision |              69.4391 |                        70.1888 |                          65.7987 |
| recall    |              68.7589 |                        67.1106 |                          64.2727 |
| f1-score  |              68.6762 |                        67.9613 |                          64.4688 |

#### Fine-grained labels

|           |   deepset/gbert-base |   dbmdz/bert-base-german-cased |   dbmdz/bert-base-german-uncased |
|:----------|---------------------:|-------------------------------:|---------------------------------:|
| precision |              63.1794 |                        58.7445 |                          57.2049 |
| recall    |              54.1478 |                        51.6786 |                          51.3879 |
| f1-score  |              56.3713 |                        53.4814 |                          52.7212 |
-->   

### Few-shot classification with Fastfit

We fine-tuned Fastfit on a full train set in each 5-fold. Regarding hyperparameters, we used the hyperparameters suggested by authors for [text classification](https://github.com/IBM/fastfit?tab=readme-ov-file#training-with-python).

```json
        {
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            max_text_length=128,
            dataloader_drop_last=False,
            num_repeats=4,
            optim="adamw_torch",
            weight_decay=0.01, 
            warmup_ratio=0.1,
            clf_loss_factor=0.1,
        }
```
<!---
#### Coarse-grained labels

|           |   deepset/gbert-base |   dbmdz/bert-base-german-cased |   dbmdz/bert-base-german-uncased |
|:----------|---------------------:|-------------------------------:|---------------------------------:|
| precision |              72.8951 |                        72.6845 |                          68.8097 |
| recall    |              65.7715 |                        64.7244 |                          64.8708 |
| f1-score  |              68.0389 |                        66.9121 |                          66.2269 |
-->   

### Results
#### Coarse-grained labels


|           |   deepset/gbert-base |   dbmdz/bert-base-german-cased |   dbmdz/bert-base-german-uncased |
|:----------|---------------------:|-------------------------------:|---------------------------------:|
| **baseline**          |                      |                    |                                  |
| precision |              68.8114 |                        67.0367 |                          70.4424 |
| recall    |              65.6223 |                        64.3296 |                          66.2102 |
| f1-score  |              66.5067 |                        65.05   |                          67.764  |
| **bestrun**          |                      |                     |                                  |
| precision |              69.4391 |                        70.1888 |                          65.7987 |
| recall    |              68.7589 |                        67.1106 |                          64.2727 |
| f1-score  |              68.6762 |                        67.9613 |                          64.4688 |
| **few-shot**          |                      |                   |                                  |
| precision |              73.9737 |                        70.0724 |                          72.3905 |
| recall    |              66.2495 |                        65.0287 |                          66.0604 |
| f1-score  |              68.4468 |                        66.3884 |                          68.153  |


#### Fine-grained labels

|           |   deepset/gbert-base |   dbmdz/bert-base-german-cased |   dbmdz/bert-base-german-uncased |
|:----------|---------------------:|-------------------------------:|---------------------------------:|
| **baseline**          |                      |                    |                                  |
| precision |              55.8016 |                        57.8166 |                          55.9067 |
| recall    |              48.3927 |                        50.7018 |                          49.8822 |
| f1-score  |              50.1383 |                        52.5486 |                          51.8392 |
| **bestrun**          |                      |                     |                                  |
| precision |              63.1794 |                        58.7445 |                          57.2049 |
| recall    |              54.1478 |                        51.6786 |                          51.3879 |
| f1-score  |              56.3713 |                        53.4814 |                          52.7212 |
| **few-shot**          |                      |                   |                                  |
| precision |              67.6762 |                        63.5798 |                          62.4564 |
| recall    |              53.1103 |                        52.4829 |                          53.1293 |
| f1-score  |              57.0432 |                        55.2921 |                          55.7986 |


### Gemini

#### TODO


## How to fine-tune a model

### Download and convert the Speech Act Dataset

Please first run the command `python3 split.py` and download the dataset from source and split it into 5-fold. The converted dataset and splits will be stored in the folder `data/`.

### Customize config.py

In `src/config.py`, please add paths to cache_dir and output_dir.

### Fine-tuning as baseline

To fine-tune a model as baseline with selected granularity (i.e. `sa_coarse` and `sa_fine_modified`) and selected model, you can run with three arguments, i.e.: 

`python3 finetuning.py baseline sa_coarse deepset/gbert-base`. 


### Fine-tuning as bestrun with defined hyperparameter

Same as with baseline, please run `funetuning.py`, but with `bestrun` argument:

`python3 finetuning.py bestrun sa_fine_modified deepset/gbert-base`. 

### Fine-tuning as few-shot classifier

To fine-tune a few-shot classifier, please run `fewshot.py` with two arguments - granularity and model:

`python3 fewshot.py sa_fine_modified deepset/gbert-base`.


### Hyperparameter search

First, you need perform a hyperparameter search, please run `hp_search.py` with a selected granularity and model:

`python3 hp_search.py sa_coarse deepset/gbert-base`.


### Fine-tuning Gemini


TODO