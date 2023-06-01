# %%
# %pip install plotly
# %pip install json
# %pip install numpy
# %pip install pandas
# %pip install sci kit-learn

workspace_dir = '/Users/sajad/projects/personal_1402/unibot'
load_pretrained = True


import collections
import copy
import gc
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm.notebook import tqdm
from transformers import (AdamW, BertConfig, BertModel, BertTokenizer,
                          get_linear_schedule_with_warmup)

# from transformers import TFBertModel, TFBertForSequenceClassification
# from transformers import glue_convert_examples_to_features

# %%
def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

is_interactive()

# %%
input_json_path = f'{workspace_dir}/datasets/intents.json'

with open(input_json_path, "r", encoding='utf8') as f:
    json_file = json.load(f)

raw_intents = []


for key, value_list in json_file.items():
    for value in value_list:
        raw_intents.append([key, value])

data = pd.DataFrame(raw_intents, columns=['label', 'text'])

data.head()

# %%
labels = list(sorted(data['label'].unique()))
labels

# %%

groupby_label = data.groupby('label')['label'].count()
if is_interactive():
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(sorted(groupby_label.index)),
        y=groupby_label.tolist(),
        text=groupby_label.tolist(),
        textposition='auto'
    ))

    fig.update_layout(
        title_text='Distribution of label within texts [DATA]',
        xaxis_title_text='Label',
        yaxis_title_text='Frequency',
        bargap=0.2,
        bargroupgap=0.2)

    fig.show()


# %%
data['label_id'] = data['label'].apply(lambda t: labels.index(t))

train, test = train_test_split(data, test_size=0.1, random_state=1, stratify=data['label'])
train, valid = train_test_split(train, test_size=0.1, random_state=1, stratify=train['label'])

train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)
test = test.reset_index(drop=True)

x_train, y_train = train['text'].values.tolist(), train['label_id'].values.tolist()
x_valid, y_valid = valid['text'].values.tolist(), valid['label_id'].values.tolist()
x_test, y_test = test['text'].values.tolist(), test['label_id'].values.tolist()

if is_interactive():
    print(train.shape)
    print(valid.shape)
    print(test.shape)


# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if is_interactive():
    print(f'device: {device}')

# %%
# general config
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16

EPOCHS = 3
EEVERY_EPOCH = 1000
LEARNING_RATE = 2e-5
CLIP = 0.0

MODEL_NAME_OR_PATH = 'HooshvareLab/bert-fa-base-uncased'
OUTPUT_PATH = f'{workspace_dir}/models/intent_classification.bin'

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# %%
label2id = {label: i for i, label in enumerate(labels)}
id2label = {v: k for k, v in label2id.items()}

if is_interactive():
    print(f'label2id: {label2id}')
    print(f'id2label: {id2label}')

# %%
# setup the tokenizer and configuration

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
config = BertConfig.from_pretrained(
    MODEL_NAME_OR_PATH, **{
        'label2id': label2id,
        'id2label': id2label,
    })

if is_interactive():
    print(config.to_json_string())

# %%
idx = np.random.randint(0, len(train))
sample_text = train.iloc[idx]['text']
sample_label = train.iloc[idx]['label']

if is_interactive():
    print(f'Sample: \n{sample_text}\n{sample_label}')

# %%
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)


if is_interactive():
    print(f'   Texts: {sample_text}')
    print(f'   Tokens: {tokenizer.convert_tokens_to_string(tokens)}')
    print(f'Token IDs: {token_ids}')

# %%
encoding = tokenizer.encode_plus(
    sample_text,
    max_length=32,
    truncation=True,
    add_special_tokens=True, # Add '[CLS]' and '[SEP]'
    return_token_type_ids=True,
    return_attention_mask=True,
    padding='max_length',
    return_tensors='pt',  # Return PyTorch tensors
)

if is_interactive():
    print(f'Keys: {encoding.keys()}\n')
    for k in encoding.keys():
        print(f'{k}:\n{encoding[k]}')

# %%
class IntentsDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Intents. """

    def __init__(self, tokenizer, text, targets=None, label_list=None, max_len=128):
        self.text = text
        self.targets = targets
        self.has_target = isinstance(targets, list) or isinstance(targets, np.ndarray)

        self.tokenizer = tokenizer
        self.max_len = max_len

        
        self.label_map = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])

        if self.has_target:
            target = self.label_map.get(str(self.targets[item]), str(self.targets[item]))

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')
        
        inputs = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }

        if self.has_target:
            inputs['targets'] = torch.tensor(target, dtype=torch.long)
        
        return inputs


def create_data_loader(x, y, tokenizer, max_len, batch_size, label_list):
    dataset = IntentsDataset(
        text=x,
        targets=y,
        tokenizer=tokenizer,
        max_len=max_len, 
        label_list=label_list)
    
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

# %%
train_data_loader = create_data_loader(train['text'].to_numpy(), train['label'].to_numpy(), tokenizer, MAX_LEN, TRAIN_BATCH_SIZE, labels)
valid_data_loader = create_data_loader(valid['text'].to_numpy(), valid['label'].to_numpy(), tokenizer, MAX_LEN, VALID_BATCH_SIZE, labels)
test_data_loader = create_data_loader(test['text'].to_numpy(), None, tokenizer, MAX_LEN, TEST_BATCH_SIZE, labels)

# %%
sample_data = next(iter(train_data_loader))
if is_interactive():
    print(sample_data.keys())
    print(sample_data['text'])
    print(sample_data['input_ids'].shape)
    print(sample_data['input_ids'][0, :])
    print(sample_data['attention_mask'].shape)
    print(sample_data['attention_mask'][0, :])
    print(sample_data['token_type_ids'].shape)
    print(sample_data['token_type_ids'][0, :])
    print(sample_data['targets'].shape)
    print(sample_data['targets'][0])

# %%
sample_test = next(iter(test_data_loader))
if is_interactive():
    print(sample_test.keys())

# %%
class IntentModel(nn.Module):

    def __init__(self, config):
        super(IntentModel, self).__init__()

        self.bert = BertModel.from_pretrained(MODEL_NAME_OR_PATH)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# %%
gc.collect()
torch.cuda.empty_cache()
pt_model = None

# %%
pt_model = IntentModel(config=config)
pt_model = pt_model.to(device)

state_dict = torch.load(f'{workspace_dir}/models/intent_classification.pt')

if load_pretrained:
    pt_model.load_state_dict(state_dict)
    model_initialized = True

if is_interactive():
    print('pt_model', type(pt_model))

# %%
# sample data output

sample_data_text = sample_data['text']
sample_data_input_ids = sample_data['input_ids']
sample_data_attention_mask = sample_data['attention_mask']
sample_data_token_type_ids = sample_data['token_type_ids']
sample_data_targets = sample_data['targets']

# available for using in GPU
sample_data_input_ids = sample_data_input_ids.to(device)
sample_data_attention_mask = sample_data_attention_mask.to(device)
sample_data_token_type_ids = sample_data_token_type_ids.to(device)
sample_data_targets = sample_data_targets.to(device)


# outputs = F.softmax(
#     pt_model(sample_data_input_ids, sample_data_attention_mask, sample_data_token_type_ids), 
#     dim=1)

outputs = pt_model(sample_data_input_ids, sample_data_attention_mask, sample_data_token_type_ids)
_, preds = torch.max(outputs, dim=1)


if is_interactive():
    print(outputs[:5, :])
    print(preds[:5])

# %%
def simple_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def acc_and_f1(y_true, y_pred, average='weighted'):
    acc = simple_accuracy(y_true, y_pred)
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average)
    return {
        "acc": acc,
        "f1": f1,
    }

def y_loss(y_true, y_pred, losses):
    y_true = torch.stack(y_true).cpu().detach().numpy()
    y_pred = torch.stack(y_pred).cpu().detach().numpy()
    y = [y_true, y_pred]
    loss = np.mean(losses)

    return y, loss


def eval_op(model, data_loader, loss_fn):
    model.eval()

    losses = []
    y_pred = []
    y_true = []

    with torch.no_grad():
        for dl in tqdm(data_loader, total=len(data_loader), desc="Evaluation... "):
            
            input_ids = dl['input_ids']
            attention_mask = dl['attention_mask']
            token_type_ids = dl['token_type_ids']
            targets = dl['targets']

            # move tensors to GPU if CUDA is available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            targets = targets.to(device)

            # compute predicted outputs by passing inputs to the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
            
            # convert output probabilities to predicted class
            _, preds = torch.max(outputs, dim=1)

            # calculate the batch loss
            loss = loss_fn(outputs, targets)

            # accumulate all the losses
            losses.append(loss.item())

            y_pred.extend(preds)
            y_true.extend(targets)
    
    eval_y, eval_loss = y_loss(y_true, y_pred, losses)
    return eval_y, eval_loss


def train_op(model, 
             data_loader, 
             loss_fn, 
             optimizer, 
             scheduler, 
             step=0, 
             print_every_step=100, 
             eval=False,
             eval_cb=None,
             eval_loss_min=np.Inf,
             eval_data_loader=None, 
             clip=0.0):
    
    model.train()

    losses = []
    y_pred = []
    y_true = []

    for dl in tqdm(data_loader, total=len(data_loader), desc="Training... "):
        step += 1

        input_ids = dl['input_ids']
        attention_mask = dl['attention_mask']
        token_type_ids = dl['token_type_ids']
        targets = dl['targets']

        # move tensors to GPU if CUDA is available
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        targets = targets.to(device)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # compute predicted outputs by passing inputs to the model
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        
        # convert output probabilities to predicted class
        _, preds = torch.max(outputs, dim=1)

        # calculate the batch loss
        loss = loss_fn(outputs, targets)

        # accumulate all the losses
        losses.append(loss.item())

        # compute gradient of the loss with respect to model parameters
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        # perform optimization step
        optimizer.step()

        # perform scheduler step
        scheduler.step()

        y_pred.extend(preds)
        y_true.extend(targets)

        if eval:
            train_y, train_loss = y_loss(y_true, y_pred, losses)
            train_score = acc_and_f1(train_y[0], train_y[1], average='weighted')

            if step % print_every_step == 0:
                eval_y, eval_loss = eval_op(model, eval_data_loader, loss_fn)
                eval_score = acc_and_f1(eval_y[0], eval_y[1], average='weighted')

                if hasattr(eval_cb, '__call__'):
                    eval_loss_min = eval_cb(model, step, train_score, train_loss, eval_score, eval_loss, eval_loss_min)

    train_y, train_loss = y_loss(y_true, y_pred, losses)

    return train_y, train_loss, step, eval_loss_min


# %%
if not load_pretrained:
    optimizer = AdamW(pt_model.parameters(), lr=LEARNING_RATE, correct_bias=False, no_deprecation_warning=True)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()

    step = 0
    eval_loss_min = np.Inf
    history = collections.defaultdict(list)


    def eval_callback(epoch, epochs, output_path):
        def eval_cb(model, step, train_score, train_loss, eval_score, eval_loss, eval_loss_min):
            statement = ''
            statement += 'Epoch: {}/{}...'.format(epoch, epochs)
            statement += 'Step: {}...'.format(step)
            
            statement += 'Train Loss: {:.6f}...'.format(train_loss)
            statement += 'Train Acc: {:.3f}...'.format(train_score['acc'])

            statement += 'Valid Loss: {:.6f}...'.format(eval_loss)
            statement += 'Valid Acc: {:.3f}...'.format(eval_score['acc'])

            print(statement)

            if eval_loss <= eval_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    eval_loss_min,
                    eval_loss))
                
                torch.save(model.state_dict(), output_path)
                eval_loss_min = eval_loss
            
            return eval_loss_min


        return eval_cb


    for epoch in tqdm(range(1, EPOCHS + 1), desc="Epochs... "):
        train_y, train_loss, step, eval_loss_min = train_op(
            model=pt_model, 
            data_loader=train_data_loader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            step=step, 
            print_every_step=EEVERY_EPOCH, 
            eval=True,
            eval_cb=eval_callback(epoch, EPOCHS, OUTPUT_PATH),
            eval_loss_min=eval_loss_min,
            eval_data_loader=valid_data_loader, 
            clip=CLIP)
        
        train_score = acc_and_f1(train_y[0], train_y[1], average='weighted')
        
        eval_y, eval_loss = eval_op(
            model=pt_model, 
            data_loader=valid_data_loader, 
            loss_fn=loss_fn)
        
        eval_score = acc_and_f1(eval_y[0], eval_y[1], average='weighted')
        
        history['train_acc'].append(train_score['acc'])
        history['train_loss'].append(train_loss)
        history['val_acc'].append(eval_score['acc'])
        history['val_loss'].append(eval_loss)

# %%
def predict(model, comments, tokenizer, max_len=128, batch_size=32):
    data_loader = create_data_loader(comments, None, tokenizer, max_len, batch_size, None)
    
    predictions = []
    prediction_probs = []

    
    model.eval()
    with torch.no_grad():
        for dl in tqdm(data_loader, position=0):
            input_ids = dl['input_ids']
            attention_mask = dl['attention_mask']
            token_type_ids = dl['token_type_ids']

            # move tensors to GPU if CUDA is available
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            
            # compute predicted outputs by passing inputs to the model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids)
            
            # convert output probabilities to predicted class
            _, preds = torch.max(outputs, dim=1)

            predictions.extend(preds)
            prediction_probs.extend(F.softmax(outputs, dim=1))

    predictions = torch.stack(predictions).cpu().detach().numpy()
    prediction_probs = torch.stack(prediction_probs).cpu().detach().numpy()

    return predictions, prediction_probs


# %%
test_texts = test['text'].to_numpy()
preds, probs = predict(pt_model, test_texts, tokenizer, max_len=128)

if is_interactive():
    print(preds.shape, probs.shape)

# %%
y_test, y_pred = [labels.index(label) for label in test['label'].values], preds


if is_interactive():
    print(f'F1: {f1_score(y_test, y_pred, average="weighted")}')
    print()
    print(classification_report(y_test, y_pred, target_names=labels))

# %%
test_texts = np.array(['چند واحد پاس کردم'])
preds, probs = predict(pt_model, test_texts, tokenizer, max_len=128)


if is_interactive():
    print(preds)
    print(probs)
# y_test, y_pred = [labels.index(label) for label in test['label'].values], preds

# print(classification_report(y_test, y_pred, target_names=labels))

# %%
def get_intention(text: str):
    texts = np.array([text])
    preds, probs = predict(pt_model, texts, tokenizer, max_len=128)

    return preds, probs


