# Automated Speech Act Classification in Offensive German Language Tweets



## Experiment setup

- **Dataset**: [German Speech Acts Dataset](https://anonymous.4open.science/r/speech-act-dataset/version_1-1_changes.md)
- **Granularity**: coarse- and modified fine-grained speech acts. 
- **Experiment methods**: 
  - fine-tuning of encoder models:
    - baseline;
    - hyperparameter search with [Ray Tune](https://docs.ray.io/en/latest/tune/index.html);
    - few-shot classification with [Fastfit](https://github.com/IBM/fastfit);
  - fine-tuning of decoder models.

- **Selected models** are following:
  - encoder models:
    - [deepset/gbert-base](https://huggingface.co/deepset/gbert-base)
    - [dbmdz/bert-base-german-cased](https://huggingface.co/dbmdz/bert-base-german-cased)
    - [dbmdz/bert-base-german-uncased](https://huggingface.co/dbmdz/bert-base-german-uncased)
  - decoder models:
    - [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini)
    - [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B).    
    - [DiscoResearch/Llama3-German-8B](https://huggingface.co/DiscoResearch/Llama3-German-8B)

- **Evaluation method**: 5-fold cross-validation, divided in 80% train split with 1567 sentences and 20% validation split with 392 sentences. For each granularity, we created individual splits.
- **Evaluation metric**: precision, recall, f1-macro. Results of each model, each granularity and each method are in the folder [`/results`](/results).




## Dataset

German Speech Acts Dataset can be found on [GitHub](https://anonymous.4open.science/r/speech-act-dataset/version_1-1_changes.md). For experiments, we used the latest version v1.1. For 5-fold cross-validation, we split dataset where sentences were shuffled and stratified in order to preserve the percentage of samples for each class in each fold and split. For coarse- and fine-grained labels, we have created individual splits (s. tables below). 


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

### Encoders: Baseline


For our baselines, we selected default hyperparameters: 

```
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


### Encoders: Hyperparameter Search

We performed a hyperparameter search on the first train and validation split used a Python library [Ray Tune](https://docs.ray.io/en/latest/tune/index.html). The goal was to maximize F1-macro of baseline models during 30 trials. After finding the best hyperparameters, we trained and evaluated a model on 5-folds. Hyperparameter space was defined as follow:

```
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

### Encoders: Few-shot classification with Fastfit

We fine-tuned Fastfit on a full train set in each 5-fold. Regarding hyperparameters, we used the hyperparameters suggested by authors for [text classification](https://github.com/IBM/fastfit?tab=readme-ov-file#training-with-python).

```
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

### Encoders: Results
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


### Decoders: Gemini

We fine-tuned [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini) with suggested hyperparameters described in [model tuning card](https://ai.google.dev/gemini-api/docs/model-tuning). The hyperparameters are:

```
        {
            epoch_count = 20,
            batch_size=8,
            learning_rate=0.001,
        }
```

### Decoders: Llama 3

For both [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B) and [Llama3-German-8B](https://huggingface.co/DiscoResearch/Llama3-German-8B), we performed Parameter Efficient Fine-Tuning (PEFT) with the Quantized Low-Rank Adaptation (QLoRA) approach. 

### Decoders: Results

#### Coarse-grained labels


|           |   Gemini 1.5 Pro |   Llama-3.2-3B |   Llama3-German-8B |
|:----------|-----------------:|---------------:|-------------------:|
| precision |          33.4161 |         64.97  |            62.6882 |
| recall    |          31.0677 |         64.047 |            62.5868 |
| f1-score  |          28.9611 |         62.562 |            61.4098 |


#### Fine-grained labels

|           |   Gemini 1.5 Pro |   Llama-3.2-3B |   Llama3-German-8B |
|:----------|-----------------:|---------------:|-------------------:|
| precision |          45.0556 |        39.6789 |            40.6161 |
| recall    |          32.93   |        41.7964 |            42.9725 |
| f1-score  |          34.2639 |        39.512  |            39.8754 |


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

**Get access to Gemini API**: Before calling the Gemini API, you need to set up your project and configure your [API key](https://aistudio.google.com/app/apikey).

How to perform fine-tuning on Colab using the Gemini API, you can see in [finetuning_gemini.ipynb](finetuning_gemini.ipynb). 

### Fine-tuning Llama3

**Get acces to Llama-3.2-3B**: You need to generate a [huggingface token](https://huggingface.co/docs/hub/security-tokens) and get access to [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B). If you have been granted access to this model log in huggingface with `'huggingface-cli login` using huggingface_token. Then, you can start the fine-tuning.

To start the fine-tuning with selected granularity (i.e. `sa_coarse` and `sa_fine_modified`) and a selected model run:
 
 `python3 finetuning_genai.py sa_fine_modified meta-llama/Llama-3.2-3B`
 
Please note that we tested two models - `meta-llama/Llama-3.2-3B` and `DiscoResearch/Llama3-German-8B`. This code may not work with other models.
 
