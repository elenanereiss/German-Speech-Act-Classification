# -*- coding: utf-8 -*-

task_names = ['sa_coarse', 'sa_fine', 'sa_fine_modified']

column_names = ['title', 'sa_coarse', 'sa_fine', 'sa_fine_modified', 'hate_speech']

cv_number = 5

data_path = {
            "full": "data/speech_act_dataset_converted_v2.json",
            "field": "German_Speech_Act_Dataset",
            }

hp = {
    "baseline": {
        "sa_coarse": {
            "Twitter/twhin-bert-large": {
                "learning_rate": 1e-05,
                "num_train_epochs": 5,
                "per_device_train_batch_size": 16,
                "weight_decay": 1e-10,
                "adam_epsilon": 1e-08,
                "seed": 25,
                "gradient_accumulation_steps": 4
            },
            "FacebookAI/xlm-roberta-large": {
                "learning_rate": 1e-05,
                "num_train_epochs": 7,
                "per_device_train_batch_size": 16,
                "weight_decay": 1e-10,
                "adam_epsilon": 1e-08,
                "seed": 25,
                "gradient_accumulation_steps": 4
            }
        },
        "sa_fine": {
            "Twitter/twhin-bert-large": {
                "learning_rate": 1e-05,
                "num_train_epochs": 7,
                "per_device_train_batch_size": 16,
                "weight_decay": 1e-10,
                "adam_epsilon": 1e-08,
                "seed": 25,
                "gradient_accumulation_steps": 4
            },
            "FacebookAI/xlm-roberta-large": {
                "learning_rate": 1e-05,
                "num_train_epochs": 7,
                "per_device_train_batch_size": 16,
                "weight_decay": 1e-10,
                "adam_epsilon": 1e-08,
                "seed": 25,
                "gradient_accumulation_steps": 4
            }
        },
        "sa_fine_modified": {
            "Twitter/twhin-bert-large": {
                "learning_rate": 2e-05,
                "num_train_epochs": 16,
                "per_device_train_batch_size": 16,
                "weight_decay": 0.01,
                "adam_epsilon": 1e-08,
                "seed": 123,
                "gradient_accumulation_steps": 4
            },
            "FacebookAI/xlm-roberta-large": {
                "learning_rate": 2e-05,
                "num_train_epochs": 16,
                "per_device_train_batch_size": 16,
                "weight_decay": 0.01,
                "adam_epsilon": 1e-08,
                "seed": 123,
                "gradient_accumulation_steps": 4
            }
        }
    },
    "bestrun": {
        "sa_coarse": {
           "deepset/gbert-base": {
                "learning_rate": 4.4252911983758425e-05,
                "num_train_epochs": 11,
                "seed": 11,
                "per_device_train_batch_size": 8,
                "weight_decay": 9.18170627685293e-06,
                "adam_epsilon": 1.5409590994174922e-07,
                "gradient_accumulation_steps": 4
            },
            "dbmdz/bert-base-german-cased": {
                "learning_rate": 3.0454584087638082e-05,
                "num_train_epochs": 14,
                "seed": 20,
                "per_device_train_batch_size": 4,
                "weight_decay": 0.00013654218115321178,
                "adam_epsilon": 1.5976491143114858e-10,
                "gradient_accumulation_steps": 2
            },
            "dbmdz/bert-base-german-uncased": {
                "learning_rate": 0.00013079602642222736,
                "num_train_epochs": 8,
                "seed": 23,
                "per_device_train_batch_size": 8,
                "weight_decay": 1.0042163846986283e-08,
                "adam_epsilon": 3.2953579231321926e-10,
                "gradient_accumulation_steps": 4
            },
        },
    "sa_fine_modified": {
            "deepset/gbert-base": {
                "learning_rate": 3.595752515411035e-05,
                "num_train_epochs": 14,
                "seed": 18,
                "per_device_train_batch_size": 16,
                "weight_decay": 0.02439453356417284,
                "adam_epsilon": 1.9221691042566072e-09,
                "gradient_accumulation_steps": 2
            },
            "dbmdz/bert-base-german-cased": {
                "learning_rate": 9.04265102693198e-05,
                "num_train_epochs": 13,
                "seed": 21,
                "per_device_train_batch_size": 16,
                "weight_decay": 2.917624340538745e-06,
                "adam_epsilon": 5.718930193556378e-07,
                "gradient_accumulation_steps": 1
            },
            "dbmdz/bert-base-german-uncased": {
                "learning_rate": 4.938213167659996e-05,
                "num_train_epochs": 15,
                "seed": 26,
                "per_device_train_batch_size": 8,
                "weight_decay": 2.086364709070262e-12,
                "adam_epsilon": 1.2191040053787884e-10,
                "gradient_accumulation_steps": 2
            }
        }
    }
}