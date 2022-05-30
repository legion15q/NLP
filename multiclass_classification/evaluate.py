import os
import torch
import train_multiclass_bert
from transformers import AutoTokenizer
import re
import configparser


def convert_sentence_into_X_sentences(sentence_, entity) -> [str]:
    size_entity = len(entity)
    start_positions = [m.start() for m in re.finditer(entity, sentence_)]
    sentences = []
    for i in start_positions:
        sentences.append(sentence_[:i] + "{X}" + sentence_[i + size_entity:])
    return sentences


def evaluate(model_path, sentence, entity) -> [str]:
    config = configparser.ConfigParser()
    labels = ["neutral", "positive", "negative", "neg-pos"]
    model = torch.load(model_path)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config.read(BASE_DIR + "\config.ini")
    tokenizer_path = str(config["CONFIG"]["model"])
    X_sentences = convert_sentence_into_X_sentences(sentence, entity)

    device = torch.device("cuda:0")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              problem_type="multi_class_classification",
                                              num_labels=4)

    full_pred = []

    for X_sentence in X_sentences:
        X_sentence = re.sub("[^A-Za-zА-Яа-я{} ]", "", X_sentence)
        X_sentence = X_sentence.lower()
        pred = train_multiclass_bert.predict(tokenizer, model, X_sentence)
        result = labels[pred]
        # print(result)

        full_pred.append(result)
    return full_pred
