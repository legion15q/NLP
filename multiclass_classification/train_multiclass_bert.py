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
import collections
import glob
from sklearn import metrics


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

    def __init__(self, model_path, tokenizer_path, model_save_path, save_top_n=3, n_classes=4, epochs=1):
        self.model = BertForSequenceClassification.from_pretrained(model_path,
                                                                   num_labels=4,
                                                                   output_attentions=False,
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
        self.loss_fn = torch.nn.MultiMarginLoss().to(self.device)

    def fit(self):
        self.model = self.model.train()
        losses = []
        correct_predictions = 0

        for data in tqdm(self.train_loader):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)
            targets = data["targets"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = torch.argmax(outputs.logits, dim=1)
            loss = self.loss_fn(outputs.logits, targets)

            correct_predictions += torch.sum(preds == targets)

            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        train_acc = correct_predictions / len(self.train_set)
        train_loss = np.mean(losses)
        return train_acc, train_loss

    def eval(self):
        self.model = self.model.eval()
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for data in self.valid_loader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                targets = data["targets"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                # print(outputs.logits)
                preds = torch.argmax(outputs.logits, dim=1)
                loss = self.loss_fn(outputs.logits, targets)
                correct_predictions += torch.sum(preds == targets)
                losses.append(loss.item())

        val_acc = correct_predictions / len(self.valid_set)
        val_loss = np.mean(losses)
        return val_acc, val_loss

    def train(self):
        best_accuracy = 0
        best_val_loss = 10
        history = collections.deque([], self.save_top_n)
        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch + 1}/{self.epochs}')
            train_acc, train_loss = self.fit()
            print(f'\nTrain loss {train_loss} accuracy {train_acc}')
            val_acc, val_loss = self.eval()
            print(f'\nVal loss {val_loss} accuracy {val_acc}')
            print('-' * 10)
            if val_acc > best_accuracy or len(history) != history.maxlen or val_loss < best_val_loss:
                print('add model')
                path = self.model_save_path + f' epoch ={epoch: 02d}-val_loss ={val_loss: .4f}-LR ={LR : .1e}' + '.pt'
                if len(history) == history.maxlen:
                    os.remove(history[0])
                    history.popleft()
                history.append(path)
                torch.save(self.model, path)
                best_accuracy = val_acc
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

    prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

    return prediction


batch_size_ = 6
LR = 1e-6
save_model_path = os.getcwd() + '/sbert2'
model_path = 'sberbank-ai/sbert_large_nlu_ru'


# model_path = 'cointegrated/rubert-tiny2'
# save_model_path = os.getcwd() + '/test_tiny.pt'


def main():
    torch.cuda.empty_cache()
    train_data = pd.read_csv('train_dataset.csv')
    valid_data = pd.read_csv('valid_dataset.csv')
    test_data = pd.read_csv('test_dataset.csv')

    # n_classes = 4--количество классов для классификации
    # model_path = 'cointegrated/rubert-tiny'

    classifier = BertClassifier(
        model_path=model_path,
        tokenizer_path=model_path,
        n_classes=4,
        epochs=5,
        model_save_path=save_model_path,
        save_top_n=3
    )

    classifier.preparation(
        X_train=list(train_data['sentence']),
        y_train=list(train_data['tonality']),
        X_valid=list(valid_data['sentence']),
        y_valid=list(valid_data['tonality'])
    )

    classifier.train()

    test_model()

    return 1


def accuracy_per_class(preds, labels):
    preds_flat = np.array(preds).flatten()
    labels_flat = np.array(labels).flatten()
    k = 0
    labels = ["neutral", "positive", "negative", "neg-pos"]
    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'\nClass: {labels[k]}')
        print(f'\nAccuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')
        k += 1


def test_model(model_name=None):
    if model_name == None:
        saved_model_path = glob.glob(os.getcwd() + '\\*.pt')[1]
    else:
        saved_model_path = os.getcwd() + '\\' + model_name + '.pt'
    print(saved_model_path)
    test_data = pd.read_csv('test_dataset.csv')
    texts = list(test_data['sentence'])
    labels = list(test_data['tonality'])
    model = torch.load(saved_model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    predictions = [predict(tokenizer, model, t) for t in texts]
    print(
        metrics.classification_report(labels, predictions, target_names=['neutral', 'positive', 'negative', 'pos-neg']))
    accuracy_per_class(predictions, labels)
    return 1


if __name__ == '__main__':
    test_model('sbert epoch = 1-val_loss = 0.6856-LR = 1.0e-06')
    # test_model('sbert_no_upsampling epoch = 2-val_loss = 0.1490-LR = 1.0e-05')
