import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datetime import datetime
import json
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report, accuracy_score

from datasets import DatasetDict, load_dataset, ClassLabel #Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from src.config import my_cache_dir, my_output_dir
from src.cross_validation import initialize_scores_dict, sum_cv_scores, mean_cv_scores
from src.custom_dataset import task_names, column_names, cv_number

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

def main():
    class CustomTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            # Ensure label_weights is a tensor
            if class_weights is not None:
                #            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.args.device) # old param -> error with new version of transformers
                self.class_weights = class_weights.type(dtype=torch.float32).clone().detach().to(self.args.device)
            else:
                self.class_weights = None

        # def compute_loss(self, model, inputs, return_outputs=False): # old param -> error with new version of transformers
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Extract labels and convert them to long type for cross_entropy
            labels = inputs.pop("labels").long()

            # Forward pass
            outputs = model(**inputs)

            # Extract logits assuming they are directly outputted by the model
            logits = outputs.get('logits')

            # Compute custom loss with class weights for imbalanced data handling
            if self.class_weights is not None:
                loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            else:
                loss = F.cross_entropy(logits, labels)

            return (loss, outputs) if return_outputs else loss
    #        return (loss(logits, labels), outputs) if return_output else loss(logits, labels) # old param -> error with new version of transformers

    def llama_preprocessing_function(examples):
        # Tokenize the text
        tokenized_examples = tokenizer(examples['text'], truncation=True, max_length=MAX_LEN)

        # Map labels to numerical values
        tokenized_examples['label'] = ClassLabels.str2int(examples[task])
        return tokenized_examples

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'accuracy': accuracy_score(predictions, labels),
                'macro f1': f1_score(predictions, labels, average="macro", zero_division=0)}


    # Input task name
    task = sys.argv[1]
    if task not in task_names:
        print("{} is an unknown task. Please choose between {}".format(task, task_names))
        exit()

    # Input model name
    checkpoint = sys.argv[2]

    # collect labels from dataset
    dataset_train = load_dataset("json", data_files="data/{}_cv0.json".format(task.replace("sa_", "")),
                                 field="train")
    labels = sorted(set(label for label in dataset_train["train"][task]))

#    num_lab = len(labels)

    # Cast to ClassLabel
    ClassLabels = ClassLabel(num_classes=len(labels), names=labels)
    col_to_delete = column_names.remove(task)

    scores_dict = {}

    print("\n\n\n****************************Training of genai {}****************************\n".format(checkpoint))

    # cross-validation
    fold = cv_number


    # save results to folder results
    file_name = "results/genai_{}_{}.txt".format(task, checkpoint.split("/")[-1])
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


        dataset = DatasetDict({
            'train': dataset_train['train'],
            'val': dataset_valid['train'],
            'test': dataset_valid['train']
        })

        # Load the JSON data
        with open("data/{}_cv{}.json".format(task.replace("sa_", ""), str(n)), 'r') as f:
            data = json.load(f)

        # Access the 'train' field and create the DataFrame
        df_train = pd.DataFrame(data['train'])
        df_test = pd.DataFrame(data['test'])
        df_valid = pd.DataFrame(data['test'])

        df_train[task]=df_train[task].astype('category')
        df_train['target']=df_train[task].cat.codes
        num_labels=len(df_train[task].cat.categories)
        category_map = {code: category for code, category in enumerate(df_train[task].cat.categories)}
        class_weights=(1/df_train.target.value_counts(normalize=True).sort_index()).tolist()
        class_weights=torch.tensor(class_weights)
        class_weights=class_weights/class_weights.sum()

        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True, # enable 4-bit quantization
            bnb_4bit_quant_type = 'nf4', # information theoretically optimal dtype for normally distributed weights
            bnb_4bit_use_double_quant = True, # quantize quantized weights //insert xzibit meme
            bnb_4bit_compute_dtype = torch.bfloat16 # optimized fp format for ML
        )

        lora_config = LoraConfig(
            r = 16, # the dimension of the low-rank matrices
            lora_alpha = 8, # scaling factor for LoRA activations vs pre-trained weight activations
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout = 0.05, # dropout probability of the LoRA layers
            bias = 'none', # wether to train bias weights, set to 'none' for attention layers
            task_type = 'SEQ_CLS'
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            quantization_config=quantization_config,
            num_labels=num_labels,
            cache_dir=my_cache_dir
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True, cache_dir=my_cache_dir)

        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.config.pretraining_tp = 1

        MAX_LEN = 512


        tokenized_datasets = dataset.map(llama_preprocessing_function, batched=True, remove_columns=col_to_delete)
        tokenized_datasets.set_format("torch")

        collate_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        training_args = TrainingArguments(
            output_dir = my_output_dir,
            learning_rate = 1e-4,
            per_device_train_batch_size = 8,
            per_device_eval_batch_size = 8,
            num_train_epochs = 5,
            weight_decay = 0.01,
            eval_strategy = 'epoch', #arg eval_strategy new for trasformers 4.46
            save_strategy = 'epoch',
            load_best_model_at_end = True
        )

        trainer = CustomTrainer(
            model = model,
            args = training_args,
            train_dataset = tokenized_datasets['train'],
            eval_dataset = tokenized_datasets['val'],
            processing_class = tokenizer, #arg processing_class new for trasformers 4.46
            data_collator = collate_fn,
            compute_metrics = compute_metrics,
            class_weights=class_weights,
        )

        train_result = trainer.train()

        def make_predictions(model,df_test):


          # Convert summaries to a list
          sentences = df_test.text.tolist()

          # Define the batch size
          batch_size = 32  # You can adjust this based on your system's memory capacity

          # Initialize an empty list to store the model outputs
          all_outputs = []

          # Process the sentences in batches
          for i in range(0, len(sentences), batch_size):
              # Get the batch of sentences
              batch_sentences = sentences[i:i + batch_size]

              # Tokenize the batch
              inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

              # Move tensors to the device where the model is (e.g., GPU or CPU)
              inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

              # Perform inference and store the logits
              with torch.no_grad():
                  outputs = model(**inputs)
                  all_outputs.append(outputs['logits'])
          final_outputs = torch.cat(all_outputs, dim=0)
          df_test['predictions']=final_outputs.argmax(axis=1).cpu().numpy()
          df_test['predictions']=df_test['predictions'].apply(lambda l:category_map[l])


        make_predictions(model,df_test)

#        get_performance_metrics(df_test)
        if task=="sa_fine_modified": valid_labels = df_test.sa_fine_modified
        else: valid_labels = df_test.sa_coarse
        predicted_labels = df_test.predictions

        cv_results = sum_cv_scores(cv_results, classification_report(valid_labels, predicted_labels, zero_division=0, digits=4, output_dict=True))

        results = classification_report(valid_labels, predicted_labels, zero_division=0, digits=4)
        print(results)

        # Write results to file
        w.write(
            "**************************************************Cross-validation number {}**************************************************\n\n".format(n + 1))
        w.write(results)
        w.write("\n")

    cv_results = mean_cv_scores(cv_results)

    output = "************************************************Results of 5 fold cross-validation************************************************\n\n"
    output += '%12s%11s%11s%11s%11s\n\n' % ('', 'precision', 'recall', 'f1-score', 'support')
    for key, value in cv_results.items():
        if key != 'accuracy':
            string = ""
            for key2, value2 in cv_results[key].items():
                if key2 != 'support':
                    string += '%*.*f' % (11, 4, value2)
                else:
                    string += '%*.*f' % (11, 2, value2)
            output += '%12s%s\n' % (key, string)
        else:
            output += '\n%12s%*.*f' % ('accuracy', 33, 4, cv_results['accuracy'])
            output += '%*.*f\n' % (11, 2, cv_results['macro avg']['support'])
    print(output)

    w.write(output)
    w.close()

    scores_dict[checkpoint] = cv_results["macro avg"]

    scores = open("results/genai_{}_scores_dict.py".format(task), "a", encoding="utf-8")
    scores.write("'{}' = {}\n\n".format(str(datetime.now()), json.dumps(scores_dict, indent=4)))
    scores.close()

    print(scores_dict)
    print(cv_results)

if __name__ == "__main__":
    main()