import collections
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from transformers import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, \
    BertForSequenceClassification
from torch.utils.data import DataLoader, SequentialSampler
import pandas as pd
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import os
import random
from tqdm import tqdm
from torch.utils.data import TensorDataset
from sklearn import metrics
import glob
import configparser
import config




class CustomDataset(Dataset):

    def __init__(self, texts, targets, tokenizer, max_len=256):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


class BertClassifier:

    def __init__(self, model_path, tokenizer_path, model_save_path, save_top_n=3, n_classes=3, epochs=1):
        self.model = BertForSequenceClassification.from_pretrained(model_path,
                                                                   num_labels=3,
                                                                   output_attentions=False,
                                                                   problem_type="multi_label_classification",
                                                                   return_dict=True,
                                                                   output_hidden_states=False)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.device = torch.device("cuda:0")  # if torch.cuda.is_available() else "cpu"
        self.model_save_path = model_save_path
        self.max_len = 256
        self.epochs = epochs
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = torch.nn.Linear(self.out_features, n_classes)
        self.model.to(self.device)
        self.save_top_n = save_top_n

    def preparation(self, X_train, y_train, X_valid, y_valid):
        # create datasets
        self.train_set = CustomDataset(X_train, y_train, self.tokenizer)
        self.valid_set = CustomDataset(X_valid, y_valid, self.tokenizer)

        # create data loaders
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size_,
                                       sampler=SequentialSampler(self.train_set))
        self.valid_loader = DataLoader(self.valid_set, batch_size=batch_size_,
                                       sampler=SequentialSampler(self.valid_set))

        # helpers initialization
        # чтобы val loss падал нужен минимум 1e-06
        self.optimizer = AdamW(self.model.parameters(),
                               lr=LR,
                               eps=1e-8)
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)

        torch.cuda.manual_seed_all(seed_val)
        warmup_steps = len(self.train_loader) // 3
        total_steps = len(self.train_loader) * self.epochs - warmup_steps
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        self.loss_fn = torch.nn.MultiLabelSoftMarginLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []

        for data in tqdm(self.train_loader):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # preds = torch.argmax(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, targets)

            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_loss = np.mean(losses)
        return train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.loss_fn(outputs.logits, targets)
                losses.append(loss.item())

        val_loss = np.mean(losses)
        return val_loss

    def train(self):
        best_val_loss = 10
        history = collections.deque([], self.save_top_n)
        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch + 1}/{self.epochs}')
            train_loss = self.fit()
            print(f'\nTrain loss {train_loss}')
            val_loss = self.eval()
            print(f'\nVal loss {val_loss}')
            print('-' * 10)
            if val_loss < best_val_loss or len(history) != history.maxlen:
                print('add model')
                path = self.model_save_path + f' epoch ={epoch: 02d}-val_loss ={val_loss: .4f}-LR ={LR : .1e}' + '.pt'
                if len(history) == history.maxlen:
                    os.remove(history[0])
                    history.popleft()
                history.append(path)
                torch.save(self.model, path)
                best_val_loss = val_loss

        self.model = torch.load(history[-1])

    def predict(self, text):
        predict(self.tokenizer, self.model, text, self.device, self.max_len)


def predict(tokenizer, model, text, device=torch.device("cuda:0"), max_len=256):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

    out = {
        'text': text,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten()
    }

    input_ids = out["input_ids"].to(device)
    attention_mask = out["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attention_mask.unsqueeze(0)
    )

    prediction = outputs.logits
    return prediction


save_model_path = os.getcwd() + '/sbert_no_upsampling'


# model_path = 'cointegrated/rubert-tiny2'
# save_model_path = os.getcwd() + '/bert_tiny2'


def main():
    torch.cuda.empty_cache()
    train_data = pd.read_csv('train_dataset.csv')
    valid_data = pd.read_csv('valid_dataset.csv')

    # n_classes = 4--количество классов для классификации
    # model_path = 'cointegrated/rubert-tiny'

    classifier = BertClassifier(
        model_path=model_path,
        tokenizer_path=model_path,
        n_classes=3,
        epochs=5,
        model_save_path=save_model_path,
        save_top_n=3,
    )

    df_tags = train_data[train_data.columns[1:]]
    y_tr = np.array(df_tags.iloc[:])
    df_tags = valid_data[valid_data.columns[1:]]
    y_val = np.array(df_tags.iloc[:])

    classifier.preparation(
        X_train=list(train_data['sentence']),
        y_train=y_tr,
        X_valid=list(valid_data['sentence']),
        y_valid=y_val,
    )

    classifier.train()

    test_model()

    return 1


def accuracy_per_class(predicted_labels, true_labels):
    labels = ["neutral", "positive", "negative", "neg-pos"]
    class_labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]])
    k = 0
    # label_count = 0
    for i in range(len(class_labels)):
        correct_preds_per_class = 0
        true_per_class = 0
        for j in range(len(predicted_labels)):
            if np.all(true_labels[j] == class_labels[i]):
                true_per_class += 1
                if np.all(predicted_labels[j] == class_labels[i]):
                    correct_preds_per_class += 1
        # label_count += 1
        print(f'Class: {labels[k]}')
        print(f'Accuracy: {correct_preds_per_class}/{true_per_class}\n')
        k += 1

def classify(pred_prob, thresh):
    y_pred = []

    for tag_label_row in pred_prob:
        temp = []
        for tag_label in tag_label_row:
            if tag_label >= thresh:
                temp.append(1)  # Infer tag value as 1 (present)
            else:
                temp.append(0)  # Infer tag value as 0 (absent)
        y_pred.append(temp)

    return y_pred

def test_model(model_name=None, posteriori_opt_threshold_=None):
    test_data = pd.read_csv('test_dataset.csv')
    texts = list(test_data['sentence'])

    df_tags = test_data[test_data.columns[1:]]
    y_test = np.array(df_tags.iloc[:])
    print(y_test)
    labels = y_test
    if model_name == None:
        saved_model_path = glob.glob(os.getcwd() + '\\*.pt')[1]
    else:
        saved_model_path = os.getcwd() + '\\' + model_name + '.pt'
    print(saved_model_path)
    model = torch.load(saved_model_path)
    device = torch.device("cuda:0")
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    input_ids = []
    attention_masks = []
    for quest in texts:
        encoded_quest = tokenizer.encode_plus(
            quest,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        # Add the input_ids from encoded question to the list.
        input_ids.append(encoded_quest['input_ids'])
        # Add its attention mask
        attention_masks.append(encoded_quest['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(y_test)

    # Set the batch size.
    TEST_BATCH_SIZE = 64
    pred_data = TensorDataset(input_ids, attention_masks, labels)
    pred_sampler = SequentialSampler(pred_data)
    pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=TEST_BATCH_SIZE)

    flat_pred_outs = 0
    flat_true_labels = 0

    model.eval()

    pred_outs, true_labels = [], []
    # i=0
    # Predict
    for batch in pred_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_attn_mask, b_labels = batch

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            pred_out = model(b_input_ids, b_attn_mask).logits
            pred_out = torch.sigmoid(pred_out)
            # Move predicted output and labels to CPU
            pred_out = pred_out.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # i+=1
            # Store predictions and true labels
            # print(i)
            # print(outputs)
            # print(logits)
            # print(label_ids)
        pred_outs.append(pred_out)
        true_labels.append(label_ids)
    # print(pred_outs)
    flat_pred_outs = np.concatenate(pred_outs, axis=0)
    flat_true_labels = np.concatenate(true_labels, axis=0)
    print(flat_pred_outs.shape, flat_true_labels.shape)
    # define candidate threshold values
    threshold = np.arange(0.1, 0.51, 0.01)

    # convert probabilities into 0 or 1 based on a threshold value


    # convert labels to 1D array
    y_true = flat_true_labels.ravel()
    scores = []
    opt_thresh = 0
    if posteriori_opt_threshold_ == None:
        for thresh in threshold:
            # classes for each threshold
            pred_bin_label = classify(flat_pred_outs, thresh)

            # convert to 1D array
            y_pred = np.array(pred_bin_label).ravel()

            scores.append(metrics.f1_score(y_true, y_pred))
        opt_thresh = threshold[scores.index(max(scores))]
    else:
        opt_thresh = posteriori_opt_threshold_
    print(f'Optimal Threshold Value = {opt_thresh}')
    file = open("Optimal_Threshold_Value_for_multilabel.txt","w+")
    file.write("[TEMP]\n")
    file.write("opt_thresh="+str(opt_thresh))
    file.close()
    y_pred_labels = classify(flat_pred_outs, opt_thresh)
    print(y_true)
    ''''''
    # отрезаем взаимоисключающие классы
    for i in range(len(flat_pred_outs)):
        if np.all(y_pred_labels[i] == [1, 0, 1]):
            arg_max = np.argmax(flat_pred_outs[i])
            if arg_max == 0:
                y_pred_labels[i] = [1, 0, 0]
            else:
                y_pred_labels[i] = [0, 0, 1]
        if np.all(y_pred_labels[i] == [1, 1, 0]):
            arg_max = np.argmax(flat_pred_outs[i])
            if arg_max == 0:
                y_pred_labels[i] = [1, 0, 0]
            else:
                y_pred_labels[i] = [0, 1, 0]
        if np.all(y_pred_labels[i] == [1, 1, 1]):
            arg_max = np.argmax(flat_pred_outs[i])
            if arg_max == 0:
                y_pred_labels[i] = [1, 0, 0]
            else:
                y_pred_labels[i] = [0, 0, 0]
                y_pred_labels[i][arg_max] = 1

    # print(y_pred_labels)
    y_pred = np.array(y_pred_labels).ravel()  # Flatten
    # print(y_pred)
    # print(metrics.classification_report(y_true, y_pred))
    y_pred_old_class = convert_to_previous_class(y_pred_labels)
    y_true_old_class = convert_to_previous_class(flat_true_labels)
    print(metrics.classification_report(y_true_old_class, y_pred_old_class,
                                        target_names=['neg-pos', 'negative', 'neutral', 'positive']))
    accuracy_per_class(y_pred_labels, flat_true_labels)

    return 1


def convert_to_previous_class(array):
    previous_class_array = []
    for i in array:
        if np.all(i == [1, 0, 0]) or np.all(i == [0,0,0]):
            previous_class_array.append(0)
        if np.all(i == [0, 1, 0]):
            previous_class_array.append(1)
        if np.all(i == [0, 0, 1]):
            previous_class_array.append(-1)
        if np.all(i == [0, 1, 1]):
            previous_class_array.append(-2)
    return previous_class_array

def parse_config():
    config = configparser.ConfigParser()
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config.read(BASE_DIR + "\config.ini")
    global batch_size_, LR, model_path, epochs_, save_top_n_
    batch_size_ = int(config["CONFIG"]["batch_size"])
    LR = float(config["CONFIG"]["LR"])
    model_path = str(config["CONFIG"]["model"])
    epochs_ = int(config["CONFIG"]["epochs"])
    save_top_n_ = int(config["CONFIG"]["save_top_n"])
    return 1

if __name__ == '__main__':
    parse_config()
    #main()
    test_model('sbert epoch = 1-val_loss = 0.6308-LR = 1.0e-06')
    # test_model('sbert epoch = 1-val_loss = 0.6308-LR = 1.0e-06')
    # test_model('sbert_no_upsampling epoch = 1-val_loss = 0.3481-LR = 1.0e-05')
