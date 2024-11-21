# -*- coding: utf-8 -*-
import os
import requests
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from src.config import dataset_link


if __name__ == "__main__":


    # Split dataset with sklearn
    
    r = requests.get(dataset_link)
    data = json.loads(r.text)
    map_fine_labels = {"ENGAGE": "COMMISSIVE", "REFUSE": "COMMISSIVE", "THREAT": "COMMISSIVE","DISAGREE": "EXCLUDED", "APOLOGIZE": "EXCLUDED", "THANK": "EXCLUDED", "GREET": "EXCLUDED"}

    # Dataframe for annotations (Title/filename, text, informativeness, topic and credibility rating for each tweet)
    df = pd.DataFrame(columns=['title', 'text', 'sa_coarse', 'sa_fine', 'sa_fine_modified', 'hate_speech'])

    # create data folder
    if not os.path.exists('data'):
        os.makedirs('data')

    with open('data/speech_act_dataset_v1_1.json', encoding="utf8") as f:
        data = json.load(f)
        titles = [k for k in data.keys()]
        for t in titles:
            hate_label = t.split("_")[-1].replace(".xml","")
            sentences = data[t]["tweet"]["sentences"]
            for sent_number in sentences.keys():
                label_coarse = sentences[sent_number]["coarse"]
                label_fine = sentences[sent_number]["fine"]
                if label_fine in map_fine_labels.keys(): 
                    sa_fine_modified = map_fine_labels[label_fine]
                else: sa_fine_modified = label_fine
                text = sentences[sent_number]["text"]
                sentence = pd.Series({'title': t, 'text': text, 'sa_coarse': label_coarse, 'sa_fine': label_fine, 'sa_fine_modified': sa_fine_modified, 'hate_speech': hate_label})
                df = pd.concat([df, sentence.to_frame().T], ignore_index=True)
                
    data = df.to_dict('records')
    # Serializing and writing pretty json file
    # important to use with encoding="utf-8" and ensure_ascii=False
    with open('data/speech_act_dataset_converted_v2.json', "w", encoding="utf-8") as f:
        new_data = {"German_Speech_Act_Dataset": data}
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    print("Dataset converted and saved to 'speech_act_dataset_converted_v2.json'")
    
    # Cast to numpy
    X=df.to_numpy()
    y_coarse=df['sa_coarse'].to_numpy()
    y_fine=df['sa_fine_modified'].to_numpy()


    # Coarse-grained version
    # Split dataset in 5-fold cv
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # Write to files
    for i, (train_index, test_index) in enumerate(sss.split(X, y_coarse)):

        cv_train = pd.DataFrame(columns=['title', 'text', 'sa_coarse', 'sa_fine', 'sa_fine_modified', 'hate_speech'])
        cv_train = cv_train.append(pd.DataFrame(X[train_index], columns=cv_train.columns), ignore_index=True)
        cv_train = cv_train.to_dict('records')

        cv_test = pd.DataFrame(columns=['title', 'text', 'sa_coarse', 'sa_fine', 'sa_fine_modified', 'hate_speech'])
        cv_test = cv_test.append(pd.DataFrame(X[test_index], columns=cv_test.columns), ignore_index=True)
        cv_test = cv_test.to_dict('records')

       # Serializing and writing pretty json file
        # important to use with encoding="utf-8" and ensure_ascii=False
        with open('data/coarse_cv{}.json'.format(i), "w", encoding="utf-8") as f:
            new_data = {"train": cv_train, "test": cv_test}
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        print("Data saved to JSON file ")


    # Fine-grained version
    # Split dataset in 5-fold cv
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # Write to files
    for i, (train_index, test_index) in enumerate(sss.split(X, y_fine)):

        cv_train = pd.DataFrame(columns=['title', 'text', 'sa_coarse', 'sa_fine', 'sa_fine_modified', 'hate_speech'])
        cv_train = cv_train.append(pd.DataFrame(X[train_index], columns=cv_train.columns), ignore_index=True)
        cv_train = cv_train.to_dict('records')

        cv_test = pd.DataFrame(columns=['title', 'text', 'sa_coarse', 'sa_fine', 'sa_fine_modified', 'hate_speech'])
        cv_test = cv_test.append(pd.DataFrame(X[test_index], columns=cv_test.columns), ignore_index=True)
        cv_test = cv_test.to_dict('records')

       # Serializing and writing pretty json file
        # important to use with encoding="utf-8" and ensure_ascii=False
        with open('data/fine_modified_cv{}.json'.format(i), "w", encoding="utf-8") as f:
            new_data = {"train": cv_train, "test": cv_test}
            json.dump(new_data, f, indent=4, ensure_ascii=False)
        print("Data saved to JSON file ")
