import json
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from datasets import load_dataset
from spacy.lang.en import English
from spacy.pipeline.dep_parser import defaultdict
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from preprocessing_pipeline import multi_file


def nlp_on_text(text_df1):
    nlp = English()
    text_df1['tokens'] = text_df1['Full_Text'].apply(lambda x: nlp(x))

    # Split tokens into a list ready for CSV
    text_df1['split_tokens'] = text_df1['tokens'].apply(lambda x: [tok.text for tok in x])

    # Create dummy NER tags for alignment purposes (a bit lazy, but convinient)
    text_df1['dummy_ner_tags'] = text_df1['tokens'].apply(lambda x: [0 for _ in x])

    # Serialise the data to JSON for archive
    export_columns = ['split_tokens', 'dummy_ner_tags']
    export_df = text_df1[export_columns]
    export_df.to_json('test1.json', orient="table", index=False)
    # Re-import the serialized JSON data and create a dataset in the format needed for the transformer
    dataset = load_dataset('json', data_files='test1.json', field='data')
    return dataset


def word_id_func(input_ids, tokenizer0, print_labs=False):
    tokens = tokenizer0.convert_ids_to_tokens(input_ids)

    word_ids = []
    i = 0
    spec_toks = ['[CLS]', '[SEP]', '[PAD]']
    for t in tokens:
        if t in spec_toks:
            word_ids.append(-100)
            print(t, i) if print_labs else None
        elif t.startswith('▁'):
            i += 1
            word_ids.append(i)
            print(t, i) if print_labs else None
        else:
            word_ids.append(i)
            print(t, i) if print_labs else None
        print("Total:", i) if print_labs else None
    return word_ids


def tokenize_and_align_labels(examples, label_all_tokens=False):
    tokenized_inputs = tokenizer(examples["split_tokens"],
                                 truncation=True,
                                 is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["dummy_ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def model_for_pred(device):
    tokenizer = transformers.RobertaTokenizerFast.from_pretrained("roberta-base", add_prefix_space=True)
    loaded_model = AutoModelForTokenClassification.from_pretrained("model_saved")
    loaded_model.to(device)
    args = TrainingArguments(output_dir="test_model",
                             per_device_train_batch_size=4,
                             per_device_eval_batch_size=4,
                             seed=37
                             )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    pred_trainer1 = Trainer(
        loaded_model,
        args,
        data_collator=data_collator,
        tokenizer=tokenizer)
    return pred_trainer1, tokenizer


'''def predict(pred_trainer, tokenized_dataset, label_list):
    predictions, labels, _ = pred_trainer.predict(tokenized_dataset["train"])
    predictions = np.argmax(predictions, axis=2)
    text_df['predictions'] = list(predictions)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    text_df['true_predictions'] = true_predictions

    predictions = np.argmax(predictions, axis=2)
    text_df['predictions'] = list(predictions)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    text_df['true_predictions'] = true_predictions'''

'''def data_extract(tuple_list):
        de_list = []
        for tup in tuple_list:
            if tup[1] != 'O':
                de_list.append(tup)
        return de_list

    text_df['check_pred'] = list(list(zip(a, b)) for a, b in zip(text_df['split_tokens'], text_df['true_predictions']))
    text_df['data_tuples'] = text_df['check_pred'].apply(data_extract)'''


def extract_agreement_date(tuple_list):
    temp_date = ''
    for d in tuple_list:
        if d[1] == "B-AGMT_DATE":
            temp_date = d[0]
        elif d[1] == "I-AGMT_DATE":
            temp_date = temp_date + " " + d[0]
        else:
            continue
    return temp_date


def extract_agreement_name(tuple_list):
    temp_name = ''
    for n in tuple_list:
        if n[1] == "B-DOC_NAME":
            temp_name = n[0]
        elif n[1] == "I-DOC_NAME":
            temp_name = temp_name + " " + n[0]
        else:
            continue
    return temp_name


def extract_agreement_parties(tuple_list):
    temp_party = ''
    data_dict = defaultdict(list)
    for i, p in enumerate(tuple_list):
        if p[1] == "B-PARTY":
            temp_party = p[0]
            if i == (len(tuple_list) - 1):
                data_dict["Parties"].append(temp_party)
            elif tuple_list[i + 1][1] != "I-PARTY":
                data_dict["Parties"].append(temp_party)
        elif p[1] == "I-PARTY":
            temp_party = temp_party + " " + p[0]
            if i == (len(tuple_list) - 1):
                data_dict["Parties"].append(temp_party)
            elif tuple_list[i + 1][1] != "I-PARTY":
                data_dict["Parties"].append(temp_party)

    return list(dict.fromkeys(data_dict['Parties']))


if __name__ == '__main__':
    accelerator = Accelerator()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_trainer, tokenizer = model_for_pred(device)
    accelerator = ()

    multi_file()
    text_df = pd.read_csv('data/text.csv')

    datasets = nlp_on_text(text_df)
    tokenized_dataset = datasets.map(tokenize_and_align_labels, batched=True)
    with open('data/feature_class_labels.json', 'r') as f:
        label_list = json.load(f)

    predictions, labels, _ = pred_trainer.predict(tokenized_dataset["train"])
    predictions = np.argmax(predictions, axis=2)
    text_df['predictions'] = list(predictions)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    text_df['true_predictions'] = true_predictions


    def data_extract(tuple_list):
        de_list = []
        for tup in tuple_list:
            if tup[1] != 'O':
                de_list.append(tup)
        return de_list


    text_df['check_pred'] = list(list(zip(a, b)) for a, b in zip(text_df['split_tokens'], text_df['true_predictions']))
    text_df['data_tuples'] = text_df['check_pred'].apply(data_extract)

    text_df['agmt_date'] = text_df['data_tuples'].apply(extract_agreement_date)
    text_df['agmt_parties'] = text_df['data_tuples'].apply(extract_agreement_parties)
    text_df['agmt_name'] = text_df['data_tuples'].apply(extract_agreement_name)
    # Create a dataframe with just the information we want to keep and

    export_df = text_df[['Short_Text', 'agmt_name', 'agmt_date', 'agmt_parties', 'Full_Text']].copy()

    print(text_df)
    print('---------------')
    print(text_df['agmt_name'])
    print(text_df['agmt_parties'])
    print(text_df['agmt_date'])
