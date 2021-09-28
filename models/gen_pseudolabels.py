# -*- coding: utf-8 -*-


import pandas as pd
import torch
from sklearn.model_selection import train_test_split

SEED = 42
torch.manual_seed(SEED)

df_3 = pd.read_csv('/content/eng-tam_3.csv')
df_5 = pd.read_csv('/content/eng-tam_5.csv')
df_7 = pd.read_csv('/content/eng-tam.csv')
df = pd.DataFrame()
df['text'] = df_3['text']
df['three'], uniq1 = pd.factorize(df_3['category'])
df['five'], uniq2 = pd.factorize(df_5['category'])
df['seven'], uniq2 = pd.factorize(df_7['category'])

train, val = train_test_split(df, test_size=0.1, random_state=42)

import torch
from torch.utils.data import Dataset, DataLoader


class MTLdataset(Dataset):

    def __init__(self, sentence, label1, label2, label3, tokenizer, max_len):
        self.sentence = sentence
        self.label1 = label1
        self.label2 = label2
        self.label3 = label3
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, item):
        sentence = str(self.sentence[item])
        label1 = self.label1[item]
        label2 = self.label2[item]
        label3 = self.label3[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        return {
            'sentences': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label1': torch.tensor(label1, dtype=torch.long),
            'label2': torch.tensor(label2, dtype=torch.long),
            'label3': torch.tensor(label3, dtype=torch.long)
        }


# train,test = sklearn.model_selection.train_test_split(df,random_state=42,test_size=0.1,shuffle=True)
# train,val = sklearn.model_selection.train_test_split(train,random_state=42,test_size=0.11,shuffle=True)#np.split(df.sample(frac=1, random_state=42),[int(.8*len(df)), int(.9*len(df))])
print('Training set size:', train.shape)
# print('Testing set size:',test.shape)
print('validation set size:', val.shape)
# test.sentiment2.value_counts()

import torch.nn as nn

pretrained_model = 'xlm-roberta-base'


class MTLmodel(nn.Module):

    def __init__(self, n_classes1, n_classes2, n_classes3):
        super(MTLmodel, self).__init__()
        self.model1 = AutoModel.from_pretrained("xlm-roberta-base", return_dict=False)
        self.model2 = AutoModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.model3 = AutoModel.from_pretrained("google/muril-base-cased", return_dict=False)
        self.drop = nn.Dropout(p=0.4)
        self.out1 = nn.Linear(self.model1.config.hidden_size, 128)
        self.out2 = nn.Linear(self.model2.config.hidden_size, 128)
        self.out3 = nn.Linear(self.model3.config.hidden_size, 128)
        self.drop1 = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()
        self.three = nn.Linear(128, n_classes1)
        self.five = nn.Linear(128, n_classes2)
        self.seven = nn.Linear(128, n_classes3)

    def forward(self, input_ids, attention_mask):
        _, pooled_output1 = self.model1(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, pooled_output2 = self.model2(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, pooled_output3 = self.model3(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output1)
        output = self.out1(output)
        output = self.relu(output)
        output1 = self.drop1(output)

        output = self.drop(pooled_output2)
        output = self.out2(output)
        output = self.relu(output)
        output2 = self.drop1(output)

        output = self.drop(pooled_output3)
        output = self.out3(output)
        output = self.relu(output)
        output3 = self.drop1(output)

        return {
            [self.three(output1), self.three(output2), self.three(output3)],
            [self.five(output1), self.five(output2), self.five(output3)],
            [self.seven(output1), self.seven(output2), self.seven(output3)]
        }


import time
import seaborn as sns
import matplotlib.pyplot as plt


def create_data_loader(df, batch_size, shuffle=True):
    ds = MTLdataset(
        sentence=df['text'].to_numpy(),
        label1=df['three'].to_numpy(),
        label2=df['five'].to_numpy(),
        label3=df['seven'].to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len)

    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=4)


def train_epoch(model, data_loader, loss_1, loss_2, loss_3, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses1 = []
    losses2 = []
    losses3 = []

    correct_predictions13 = 0
    correct_predictions15 = 0
    correct_predictions17 = 0
    correct_predictions23 = 0
    correct_predictions25 = 0
    correct_predictions27 = 0
    correct_predictions33 = 0
    correct_predictions35 = 0
    correct_predictions37 = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels1 = data['label1'].to(device)
        labels2 = data['label2'].to(device)
        labels3 = data['label3'].to(device)

        out1, out2, out3 = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        m13, m23, m33 = out1
        m15, m25, m35 = out2
        m17, m27, m37 = out3

        _, preds1_model1 = torch.max(m13, dim=1)
        _, preds2_model1 = torch.max(m15, dim=1)
        _, preds3_model1 = torch.max(m17, dim=1)

        _, preds1_model2 = torch.max(m23, dim=1)
        _, preds2_model2 = torch.max(m25, dim=1)
        _, preds3_model2 = torch.max(m27, dim=1)

        _, preds1_model3 = torch.max(m33, dim=1)
        _, preds2_model3 = torch.max(m35, dim=1)
        _, preds3_model3 = torch.max(m37, dim=1)

        loss13 = loss_1(m13, labels1)
        loss15 = loss_2(m15, labels2)
        loss17 = loss_3(m17, labels3)

        loss23 = loss_1(m23, labels1)
        loss25 = loss_2(m25, labels2)
        loss27 = loss_3(m27, labels3)

        loss33 = loss_1(m33, labels1)
        loss35 = loss_2(m35, labels2)
        loss37 = loss_3(m37, labels3)

        alpha = 0.4
        beta = 0.3
        gamma = 0.3

        loss_model1 = alpha * loss13 + beta * loss15 + gamma * loss17
        loss_model2 = alpha * loss23 + beta * loss25 + gamma * loss27
        loss_model3 = alpha * loss33 + beta * loss35 + gamma * loss37

        correct_predictions13 += torch.sum(preds1_model1 == labels1)
        correct_predictions15 += torch.sum(preds2_model1 == labels2)
        correct_predictions17 += torch.sum(preds3_model1 == labels3)
        correct_predictions23 += torch.sum(preds1_model2 == labels1)
        correct_predictions25 += torch.sum(preds2_model2 == labels2)
        correct_predictions27 += torch.sum(preds3_model2 == labels3)
        correct_predictions33 += torch.sum(preds1_model3 == labels1)
        correct_predictions35 += torch.sum(preds2_model3 == labels2)
        correct_predictions37 += torch.sum(preds3_model3 == labels3)

        losses1.append(loss_model1.item())
        losses2.append(loss_model2.item())
        losses3.append(loss_model3.item())

        loss13.backward(retain_graph=True)
        loss15.backward(retain_graph=True)
        loss17.backward(retain_graph=True)
        loss23.backward(retain_graph=True)
        loss25.backward(retain_graph=True)
        loss27.backward(retain_graph=True)
        loss33.backward(retain_graph=True)
        loss35.backward(retain_graph=True)
        loss37.backward(retain_graph=True)

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return {correct_predictions13.double() / n_examples,
            correct_predictions15.double() / n_examples,
            correct_predictions17.double() / n_examples,
            correct_predictions23.double() / n_examples,
            correct_predictions25.double() / n_examples,
            correct_predictions27.double() / n_examples,
            correct_predictions33.double() / n_examples,
            correct_predictions35.double() / n_examples,
            correct_predictions37.double() / n_examples}


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def show_confusion_matrix(confusion_matrix):
    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
    plt.ylabel('True label')
    plt.xlabel('predicted label')


def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    correct_predictions13 = 0
    correct_predictions15 = 0
    correct_predictions17 = 0
    correct_predictions23 = 0
    correct_predictions25 = 0
    correct_predictions27 = 0
    correct_predictions33 = 0
    correct_predictions35 = 0
    correct_predictions37 = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            labels1 = d['label1'].to(device)
            labels2 = d['label2'].to(device)
            labels3 = d['label3'].to(device)

            out1, out2, out3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            m13, m23, m33 = out1
            m15, m25, m35 = out2
            m17, m27, m37 = out3

            _, preds1_model1 = torch.max(m13, dim=1)
            _, preds2_model1 = torch.max(m15, dim=1)
            _, preds3_model1 = torch.max(m17, dim=1)

            _, preds1_model2 = torch.max(m23, dim=1)
            _, preds2_model2 = torch.max(m25, dim=1)
            _, preds3_model2 = torch.max(m27, dim=1)

            _, preds1_model3 = torch.max(m33, dim=1)
            _, preds2_model3 = torch.max(m35, dim=1)
            _, preds3_model3 = torch.max(m37, dim=1)

            correct_predictions13 += torch.sum(preds1_model1 == labels1)
            correct_predictions15 += torch.sum(preds2_model1 == labels2)
            correct_predictions17 += torch.sum(preds3_model1 == labels3)
            correct_predictions23 += torch.sum(preds1_model2 == labels1)
            correct_predictions25 += torch.sum(preds2_model2 == labels2)
            correct_predictions27 += torch.sum(preds3_model2 == labels3)
            correct_predictions33 += torch.sum(preds1_model3 == labels1)
            correct_predictions35 += torch.sum(preds2_model3 == labels2)
            correct_predictions37 += torch.sum(preds3_model3 == labels3)

    return {correct_predictions13.double() / n_examples,
            correct_predictions15.double() / n_examples,
            correct_predictions17.double() / n_examples,
            correct_predictions23.double() / n_examples,
            correct_predictions25.double() / n_examples,
            correct_predictions27.double() / n_examples,
            correct_predictions33.double() / n_examples,
            correct_predictions35.double() / n_examples,
            correct_predictions37.double() / n_examples}


"""Training!!!!!!"""

from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", return_dict=False)  # ,use_fast=False)

model = AutoModel.from_pretrained("xlm-roberta-base", return_dict=False)  # use_fast=False)

from transformers import AdamW, get_linear_schedule_with_warmup, AutoModel, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PRE_TRAINED_MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False)

BATCH_SIZE = 32
MAX_LEN = 128
train_data_loader = create_data_loader(train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(val, tokenizer, MAX_LEN, BATCH_SIZE, shuffle=False)

bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

model = MTLmodel(3, 5, 7)
model = model.to(device)

EPOCHS = 8
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
loss_1 = nn.CrossEntropyLoss().to(device)
loss_2 = nn.CrossEntropyLoss().to(device)
loss_3 = nn.CrossEntropyLoss().to(device)

from collections import defaultdict
import torch

history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
    start_time = time.time()
    train_acc1, train_acc2 = train_epoch(
        model,
        train_data_loader,
        loss_1,
        loss_2,
        loss_3,
        optimizer,
        device,
        scheduler,
        train.shape[0]
    )

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'Train Acc1 {train_acc1} Train Acc2 {train_acc2}')
    print()

    history['train_acc1'].append(train_acc1)
    history['train_acc2'].append(train_acc2)

    val_acc1, val_acc2 = eval_model(
        model,
        val_data_loader,
        device,
        val.shape[0]
    )
    print(f'Val Acc1 {val_acc1} Val Acc2 {val_acc2}')


def get_predictions(model, data_loader):
    model = model.eval()
    sentence = []

    predictions1 = []
    predictions2 = []

    prediction_probs1 = []
    prediction_probs2 = []

    real_values1 = []
    real_values2 = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["sentences"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels1 = d["label1"].to(device)
            labels2 = d["label2"].to(device)
            labels3 = d['label3'].to(device)
            out1, out2, out3 = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            m13, m23, m33 = out1
            m15, m25, m35 = out2
            m17, m27, m37 = out3

            _, preds1_model1 = torch.max(m13, dim=1)
            _, preds2_model1 = torch.max(m15, dim=1)
            _, preds3_model1 = torch.max(m17, dim=1)

            _, preds1_model2 = torch.max(m23, dim=1)
            _, preds2_model2 = torch.max(m25, dim=1)
            _, preds3_model2 = torch.max(m27, dim=1)

            _, preds1_model3 = torch.max(m33, dim=1)
            _, preds2_model3 = torch.max(m35, dim=1)
            _, preds3_model3 = torch.max(m37, dim=1)

        sentence.extend(texts)

        predictions1.extend(pred1)
        prediction_probs1.extend(out1)
        real_values1.extend(labels1)

        predictions2.extend(pred2)
        prediction_probs2.extend(out2)
        real_values2.extend(labels2)

    predictions1 = torch.stack(predictions1).cpu()
    prediction_probs1 = torch.stack(prediction_probs1).cpu()
    real_values1 = torch.stack(real_values1).cpu()

    predictions2 = torch.stack(predictions2).cpu()
    prediction_probs2 = torch.stack(prediction_probs2).cpu()
    real_values2 = torch.stack(real_values2).cpu()

    return sentence, predictions1, prediction_probs1, real_values1, predictions2, prediction_probs2, real_values2


y_review_texts, y_pred1, y_pred_probs1, y_test1, _, _, _ = get_predictions(
    model,
    test_data_loader
)
y_review_texts, _, _, _, y_pred2, y_pred_probs2, y_test2 = get_predictions(
    model,
    test_data_loader
)

from sklearn.metrics import classification_report

print(classification_report(y_test1, y_pred1, target_names=uniq1, zero_division=0))

from sklearn.metrics import classification_report

print(classification_report(y_test2, y_pred2, target_names=uniq2, zero_division=0))

"""*Generate Pseudo-labels for the best performing model*"""

df1 = pd.read_csv('/content/Trans_eng_tam.csv')
y_texts, y_pred1 = get_predictions(model, test_data_loader)
