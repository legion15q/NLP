import os
import torch
import train_multilabel_bert
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
    config.read(os.getcwd() + "/Optimal_Threshold_Value_for_multilabel.txt")
    opt_thresh = float(config["TEMP"]["opt_thresh"])
    labels = ["neutral", "positive", "negative", "neg-pos"]
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config.read(BASE_DIR + "\config.ini")
    tokenizer_path = str(config["CONFIG"]["model"])
    model = torch.load(model_path)
    X_sentences = convert_sentence_into_X_sentences(sentence, entity)

    device = torch.device("cuda:0")
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                              problem_type="multi_label_classification",
                                              num_labels=3)

    full_pred = []

    for X_sentence in X_sentences:
        X_sentence = re.sub("[^A-Za-zА-Яа-я{} ]", "", X_sentence)
        X_sentence = X_sentence.lower()
        pred = train_multilabel_bert.predict(tokenizer, model, X_sentence)
        y_pred_labels = train_multilabel_bert.classify(pred, opt_thresh)[0]
        result = ""
        if y_pred_labels == [1, 0, 0] or y_pred_labels == [0, 0, 0]:
            result = labels[0]
        elif y_pred_labels == [0, 1, 0] or y_pred_labels == [1, 1, 0]:
            result = labels[1]
        elif y_pred_labels == [0, 0, 1] or y_pred_labels == [1, 0, 1]:
            result = labels[2]
        elif y_pred_labels == [0, 1, 1] or y_pred_labels == [1, 1, 1]:
            result = labels[3]
        full_pred.append(result)
    return full_pred
