import os
import numpy as np
import pandas as pd
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, RobertaModel, \
    RobertaTokenizer, AlbertModel, AlbertTokenizer, AutoTokenizer, AutoModel
import torch
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import RandomOverSampler
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import time

from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from numpy import mean
import pickle

from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
"""Lenght of the input sentence"""
MAX_LEN = 160
"""Dropout probability, Epochs, Batch size, and Learning rate"""
DROPOUT_PROBABILITY = 0.3
EPOCHS_LIST = [3] # 10, 20, 30, 40
BATCH_SIZE_LIST = [32]  # 24, 32, 64, 128
LEARNING_RATE_LIST = [2e-5]  # 2e-5, 3e-5, 4e-5, 5e-5

## For MLP and BiLSTM
input_size = 384
hidden_size = 768
ST_model = "sentence-transformers/paraphrase-mpnet-base-v2"

"""Large Language Models"""
PRE_TRAINED_MODELS = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


optional_reqs = ['R30', 'R31', 'R32', 'R33', 'R34', 'R35', 'R36', 'R37', 'R38', 'R39', 'R40', 'R41', 'R42', 'R43', 'R44', 'R45', 'R46']
excluded_reqs = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R1XX', 'Full', 'Partial']
excluded_more_reqs = ['R24', 'R22', 'R29', 'R11', 'R13', 'R20', 'R18', 'R19', 'R21', 'R16']
reqs_list = ['R10', 'R11', 'R12', 'R13', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29']
label_names_list = ['R10', 'R11', 'R12', 'R13', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20', 'R21', 'R22', 'R23', 'R24', 'R25', 'R26', 'R27', 'R28', 'R29', 'other']
labels_list = [19, 2, 4, 16, 17, 12, 6, 15, 0, 13, 3, 1, 10, 7, 14, 5, 11, 18, 9, 8]


class Bert_Classifier(nn.Module):

  """This class defines the BERT model"""

  def __init__(self, n_classes, model_name):
      super(Bert_Classifier, self).__init__()
      self.bert = BertModel.from_pretrained(model_name, return_dict=False)
      self.drop = nn.Dropout(p=DROPOUT_PROBABILITY)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
      _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
      )
      output = self.drop(pooled_output)
      return self.out(output)


class Roberta_Classifier(nn.Module):

  """This class defines the RoBERTa model"""

  def __init__(self, n_classes, model_name):
      super(Roberta_Classifier, self).__init__()
      self.bert = RobertaModel.from_pretrained(model_name, return_dict=False)
      self.drop = nn.Dropout(p=DROPOUT_PROBABILITY)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
      _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
      )
      output = self.drop(pooled_output)
      return self.out(output)

class Albert_Classifier(nn.Module):

  """This class defines the ALBERT model"""

  def __init__(self, n_classes, model_name):
      super(Albert_Classifier, self).__init__()
      self.bert = AlbertModel.from_pretrained(model_name, return_dict=False)
      self.drop = nn.Dropout(p=DROPOUT_PROBABILITY)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
      _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
      )
      output = self.drop(pooled_output)
      return self.out(output)


class LBert_Classifier(nn.Module):

  """This class defines the Legal-BERT model"""

  def __init__(self, n_classes, model_name):
      super(LBert_Classifier, self).__init__()
      self.bert = AutoModel.from_pretrained(model_name, return_dict=False)
      self.drop = nn.Dropout(p=DROPOUT_PROBABILITY)
      self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
      _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
      )
      output = self.drop(pooled_output)
      return self.out(output)


class DPA_Dataset(Dataset):
    """This class defines a Dataset object for our dataset"""
    def __init__(self, data):
        # Initialize data, download, etc.
        # read with numpy or pandas
        self.x_data = torch.from_numpy(data['Embeddings'].apply(pd.Series).values).float()
        self.y_data = torch.from_numpy(data['label'].values).float()
        self.n_samples = data.shape[0]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



class NeuralNet(nn.Module):
  """This defines a fully connected neural network with one hidden layer"""
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.input_size = input_size
    self.l1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = self.l1(x)
    out = self.relu(out)
    out = self.l2(out)

  # no activation and no softmax at the end
    return out


class BiLSTM(nn.Module):
  """This defines a BiLSTM model with two hidden layers"""
  # define all the layers used in model
  def __init__(self, input_size, hidden_dim1, hidden_dim2, output_dim, n_layers,
          bidirectional, dropout):
    # Constructor
    super().__init__()

    # lstm layer
    self.lstm = nn.LSTM(input_size,
                      hidden_dim1,
                      num_layers=n_layers,
                      bidirectional=bidirectional,
                      batch_first=True)
    self.fc1 = nn.Linear(hidden_dim1 * 2, hidden_dim2)
    self.fc2 = nn.Linear(hidden_dim2, output_dim)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(dropout)
    # activation function
    self.act = nn.Softmax()  # \ F.log_softmax(outp)

  def forward(self, text_embedded):
    packed_output, (hidden, cell) = self.lstm(text_embedded)

    # concat the final forward and backward hidden state
    cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

    rel = self.relu(cat)
    dense1 = self.fc1(rel)

    drop = self.dropout(dense1)
    preds = self.fc2(drop)

    return preds



class GDPRReqDataset(Dataset):

    def __init__(self, sentences, targets, tokenizer, max_len):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt')

        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size, cw=False):
    ds = GDPRReqDataset(sentences=df.Sentence.to_numpy(), targets=df.label.to_numpy(), tokenizer=tokenizer,
                        max_len=max_len)
    if cw:
        labels, counts = np.unique(df['target'], return_counts=True)
        class_weights = [sum(counts) / c for c in counts]
        print(f'Class weights: {class_weights}')
        rand_weights = [class_weights[e] for e in df['label']]
        rand_sampler = WeightedRandomSampler(rand_weights, len(df['target']))
        return DataLoader(ds, batch_size=batch_size, num_workers=2, sampler=rand_sampler)
    else:
        return DataLoader(ds, batch_size=batch_size, num_workers=2, shuffle=True)


def create_directory(directory, parent_dir):
    # Path
    dir_path = os.path.join(parent_dir, directory)
    # Create the directory
    try:
        os.mkdir(dir_path)
    except OSError as error:
        print(error)


def load_dataset(df):
    # Dataset without the metadata requirements
    df.loc[df['target'].isin(excluded_reqs), 'target'] = 'other'
    # Dataset without the optional requirements
    df.loc[df['target'].isin(optional_reqs), 'target'] = 'other'
    df['target'] = pd.Categorical(df['target'])
    df['label'] = df['target'].cat.codes
    train = df.loc[df['dataset_type'] == 'train']
    test = df.loc[df['dataset_type'] == 'test']
    print(f'Train dataset size: {train.shape[0]}, Test dataset size: {test.shape[0]}')
    print(f'Total labels in train dataset: {len(train.target.unique())}')
    print(f'Total labels in test dataset {len(test.target.unique())}')
    # train = under_sample_other_label(train)
    return train, test


def load_augmented_dataset(df, df_aug):
    df_aug['label'] = df.label.loc[df.target == 'other']
    for _, row in df.iterrows():
        df_aug.loc[df_aug.target == row['target'], 'label'] = row['label']
    df_aug = df_aug[['ID', 'Sentence', 'target', 'label']]
    # print(df_aug.target.value_counts())
    # print(df_aug.label.value_counts())
    return df_aug


def under_sample_other_label(train, sample_size=0):
    # Select a fraction of others sentences from the train dataset
    df_reqs = train.loc[~train['target'].isin(['other'])]
    df_other = train.loc[train['target'].isin(['other'])]
    print(f'Number of satisfied sentences: {df_reqs.shape[0]}')
    print(f'Number of others sentences: {df_other.shape[0]}')
    # df_other = df_other.sample(n=sample_size + int((0.35 * sample_size)))  # n=2871
    df_other = df_other.sample(n=sample_size)
    print(f'Size of the sample selected from other sentences: {df_other.shape[0]}')
    df_ds = df_reqs.append(df_other, ignore_index=True)
    print(f'Size of the train dataset after sampling: {df_ds.shape[0]}')
    df_ds = shuffle(df_ds)
    return df_ds


def get_augmented_dataset_for_each_req(df_train, df_aug, req_id):
    print('============================================')
    print('Binary classification dataset')
    df_req = df_aug.loc[df_aug['target'] == req_id]
    df_req['label'] = 1
    df_ds = df_train[['ID', 'Sentence', 'target', 'label']].append(df_req, ignore_index=True)
    print(f'Total instances of the selected requirement in augmented dataset {req_id}: {df_req.shape[0]}')
    print(f'Size of the train dataset after sampling: {df_ds.shape[0]}')
    print('============================================')
    df_ds = shuffle(df_ds)
    return df_ds


def get_augmented_dataset_for_all_reqs(df_train, df_aug):
    print('============================================')
    df_all_samples = pd.DataFrame()
    df = df_train.loc[~df_train.target.isin(['other'])]
    for req_id in df.target.unique():
        df_req = df_train.loc[df_train['target'] == req_id]
        df_req_aug = df_aug.loc[df_aug['target'] == req_id]
        if len(df_req) >= 400:
            continue
        if len(df_req) > 67:
            N = 400 - len(df_req)
            df_sample = df_req_aug.sample(n=N)
            df_all_samples = df_all_samples.append(df_sample, ignore_index=True)
        else:
            df_all_samples = df_all_samples.append(df_req_aug, ignore_index=True)

    df_ds = df_train[['ID', 'Sentence', 'target', 'label']].append(df_all_samples, ignore_index=True)
    print(f'Size of the train dataset after sampling: {df_ds.shape[0]}')
    print('============================================')
    df_ds = shuffle(df_ds)
    return df_ds


def get_augmented_dataset_for_MCC(df_train, df_aug):
    print('============================================')
    print('Multi-class classification dataset')
    df_ds = df_train[['ID', 'Sentence', 'target', 'label']].append(df_aug, ignore_index=True)
    print(f'Size of original train dataset : {df_train.shape[0]}')
    print(f'Size of the augmented train dataset : {df_aug.shape[0]}')
    print(f'Size of the train dataset after oversampling: {df_ds.shape[0]}')
    print('============================================')
    df_ds = shuffle(df_ds)
    return df_ds


def under_sample_for_binary_classification(train, req_id, size_req_aug=0):
    # Select a fraction of others sentences from the train dataset
    # df_all_reqs = train.loc[~train['target'].isin(['other'])]
    # df_other = train.loc[train['target'].isin(['other'])]
    print('============================================')
    print('Binary classification dataset')
    # df_req = df_all_reqs.loc[df_all_reqs['target'] == req_id]
    df_req = train.loc[train['target'] == req_id]
    df_other = train.loc[~train['target'].isin([req_id])]
    print(f'Total instances of the selected requirement {req_id}: {df_req.shape[0]}')
    print(f'Number of others sentences: {df_other.shape[0]}')
    # df_other = df_other.sample(n=len(df_req) + (int(len(df_req)*0.5)) + size_req_aug)  # n=2871
    # df_remaining = df_all_reqs.loc[~df_all_reqs['target'].isin([req_id])]
    # print(f'Size of dataset with remaining requirements {len(df_remaining)}')
    # df_sample = df_other.sample(len(df_req) + (int(len(df_req)*0.7)) + size_req_aug)
    print(f'Original size {len(df_req)} and augmented data size {size_req_aug}')
    selected_req_20_prt = len(df_req) * 0.2
    other_20_prt = (len(df_req) + size_req_aug) * 0.2
    difference_value = other_20_prt - selected_req_20_prt
    # print(selected_req_20_prt, other_20_prt, difference_value)
    df_sample = df_other.sample(len(df_req) + int(size_req_aug) + int(difference_value))
    # df_sample = df_other.sample(len(df_req)*3)
    df_sample['target'] = 'other'
    df_sample['label'] = 0
    df_req['label'] = 1
    df_ds = df_req.append(df_sample, ignore_index=True)
    print(f'Size of the train dataset after sampling: {df_ds.shape[0]}')
    print('============================================')
    df_ds = shuffle(df_ds)
    return df_ds


def get_test_dataset_each_req(df, req_id):
    df.loc[~df['target'].isin([req_id]), 'target'] = 'other'
    df['label'] = 0
    df.loc[df['target'] == req_id, 'label'] = 1
    print(f"Size of the testing dataset {df.shape[0]}")
    print('============================================')
    return df


def get_train_dataset_each_req(df1, req_id):
    df1['target'] = df1['target'].astype('str')
    df1.loc[~df1['target'].isin([req_id]), 'target'] = 'other'
    df1.loc[~df1['target'].isin([req_id]), 'label'] = 0
    # df['label'] = 0
    df1.loc[df1['target'] == req_id, 'label'] = 1
    # print(f"Size of the training dataset {df1.shape[0]}")
    # print('============================================')
    return df1


def get_test_binary_dataset(df1):
    df1['target'] = df1['target'].astype('str')
    df1.loc[~df1['target'].isin(['other']), 'target'] = 'reqs'
    df1.loc[~df1['target'].isin(['other']), 'label'] = 1
    df1.loc[df1['target'].isin(['other']), 'label'] = 0
    print(f"Size of the testing dataset {df1.shape[0]}")
    print('============================================')
    return df1



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf
        self.max_f_score = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def early_stop_f_score(self, f_score):
        if f_score > self.max_f_score:
            self.max_f_score = f_score
            self.counter = 0
        elif f_score < (self.max_f_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def extract_results_from_report(report, model_type):
    precision, recall, f1score, f2score = 0.0, 0.0, 0.0, 0.0

    if model_type == 'binary':
        if report.iloc[1][1] == 0:
            f2score = 0
        else:
            precision = report.iloc[1][0]
            recall = report.iloc[1][1]
            f1score = report.iloc[1][2]
            f2score = (5 * precision * recall) / (4 * precision + recall)
    else:
        report = report.tail(2)
        precision = report.iloc[0][0]
        recall = report.iloc[0][1]
        f1score = report.iloc[0][2]
        f2score = (5 * precision * recall) / (4 * precision + recall)

    return precision, recall, f1score, f2score


def get_over_sampled_dataset_imblearn(df_X, df_y):
    ros = RandomOverSampler()
    X_ros, y_ros = ros.fit_sample(df_X, df_y)
    print(X_ros.shape[0] - df_X.shape[0], 'new random picked points')
    shuffle(X_ros)
    X_ros = pd.DataFrame(X_ros, columns=list(df_X.columns))
    print(X_ros.shape)
    X_ros = shuffle(X_ros)
    return X_ros

def get_test_dataset_each_req(df1, req_id):
    df1['target'] = df1['target'].astype('str')
    df1.loc[~df1['target'].isin([req_id]), 'target'] = 'other'
    df1.loc[~df1['target'].isin([req_id]), 'label'] = 0
    # df['label'] = 0
    df1.loc[df1['target'] == req_id, 'label'] = 1
    # print(f"Size of the training dataset {df1.shape[0]}")
    # print('============================================')
    return df1


def get_predictions_test(model, test_set, labels_name, labels, data_loader, parent_dir, mydevice, model_type='binary'):
    model = model.eval()
    sentence_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    correct_predictions = 0
    with torch.no_grad():
        for d in data_loader:
            texts = d["sentence_text"]
            input_ids = d["input_ids"].to(mydevice)
            attention_mask = d["attention_mask"].to(mydevice)
            targets = d["targets"].to(mydevice)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            probs = F.softmax(outputs, dim=1)
            sentence_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)
    accuracy = correct_predictions.double() / len(test_set)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    if model_type != 'binary':
        labels_name.sort()
        labels.sort()
    cm = confusion_matrix(real_values, predictions, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels_name, columns=labels_name)
    df_cm.to_excel(f'{parent_dir}/confusion_matrix.xlsx') 


def get_predictions_binary_test(model, test_set, labels_name, labels, data_loader, parent_dir, mydevice, model_type='binary'):
    model = model.eval()
    sentence_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    correct_predictions = 0
    df_result = pd.DataFrame()
    with torch.no_grad():
        for d in data_loader:
            texts = d["sentence_text"]
            input_ids = d["input_ids"].to(mydevice)
            attention_mask = d["attention_mask"].to(mydevice)
            targets = d["targets"].to(mydevice)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            probs = F.softmax(outputs, dim=1)
            sentence_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)
    accuracy = correct_predictions.double() / len(test_set)
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    if model_type != 'binary':
        labels_name.sort()
        labels.sort()
    cm = confusion_matrix(real_values, predictions, labels=labels)
    df = pd.DataFrame(cm, index=labels_name, columns=labels_name)
    df_result = df_result.append({
                'TN': df.iloc[0][0], 'FP': df.iloc[0][1],
                'TP': df.iloc[1][1],'FN': df.iloc[1][0]}, ignore_index=True)
    return df_result


def compute_single_table_for_all_DPAs(df_final, results_path):
    col_list = ['Requirement', 'TP', 'FP', 'FN', 'TN']
    df_results = pd.DataFrame()
    for req in df_final.Requirement.unique():
        df_req = df_final.loc[df_final.Requirement == req]
        df_results = df_results.append({
            'Requirement': req,
            'TP': df_req.TP.sum(),
            'FP': df_req.FP.sum(),
            'FN': df_req.FN.sum(),
            'TN': df_req.TN.sum()}, ignore_index=True)
    df_results[col_list].to_excel(f'{results_path}/all_results.xlsx', index=None)


def combine_all_DPAs_results(parent_dir):
    df_final = pd.DataFrame()
    for dir_name in os.listdir(parent_dir):
        print(dir_name)
        dpa = compute_result_per_instance_per_DPA_MCC(parent_dir+dir_name)
        df_final = pd.concat([df_final, dpa], axis=0)
    return df_final 


def compute_result_per_instance_per_DPA_MCC(parent_dir):
  df_final = pd.DataFrame()
  for root, subdirectories, files in os.walk(parent_dir):
    for subdirectory in subdirectories:
      # print(os.path.join(root, subdirectory))
      f1 = os.path.join(root, subdirectory)
      for filename in os.listdir(os.path.join(root, subdirectory)):
        df_result = pd.DataFrame()
        f = os.path.join(os.path.join(root, subdirectory), filename)
        # print(f)
        if os.path.isfile(f):
          if filename.endswith('xlsx') or filename.endswith('xls'):
            if filename.split('_')[0] == 'confusion':
              df = pd.read_excel(f)
              for req in df['Unnamed: 0'].unique():
              # for req in reqs_list:
                df.loc[df[req] > 1, req] = 1
              for req, i in zip(df['Unnamed: 0'].unique(), range(0, 20)):
                row = df.loc[df['Unnamed: 0'] == req]
                FNs = row.sum(axis=1)
                FPs = df[req].sum()
                if df[req][i] > 0:
                    TP, FP, TN, FN = 1.0, 0.0, 0.0, 0.0
                elif (FPs > 0) & (FNs.item() > 0): # FP  and FN
                    TP, FP, TN, FN = 1.0, 0.0, 0.0, 0.0
                elif (FPs == 0) & (FNs.item() > 0): # FP and FN
                    TP, FP, TN, FN = 0.0, 0.0, 0.0, 1.0
                elif (FPs > 0) & (FNs.item() == 0): # FP and FN
                    TP, FP, TN, FN = 0.0, 1.0, 0.0, 0.0
                elif (FPs == 0) & (FNs.item() == 0): # FP and FN
                    TP, FP, TN, FN = 0.0, 0.0, 1.0, 0.0
                df_result = df_result.append({
                    'DPA': f1.split('/')[4],
                    'Requirement': req,
                    'TN': TN, 'FP': FP, 'TP': TP,
                    'FN': FN }, ignore_index=True)
        col_list = ['DPA', 'Requirement', 'TP', 'FP', 'FN', 'TN']
        if len(df_result) > 0:
          df_result.sort_values(by=['Requirement'], ascending=True, inplace=True)
          df_result = df_result.loc[df_result.Requirement.isin(reqs_list)]
          TPs = df_result['TP'].sum()
          TNs = df_result['TN'].sum()
          FPs = df_result['FP'].sum()
          FNs = df_result['FN'].sum()
          total_pre = 0 if (TPs == 0) & (FPs == 0) else TPs/(TPs + FPs)
          total_rec = 0 if (TPs == 0) & (FNs == 0) else TPs/(TPs + FNs)
          total_fscore = 0 if (total_pre == 0) & (total_rec == 0) else (5*total_pre*total_rec)/(4*total_pre + total_rec)
          print(f"Results of DPA {f1.split('/')[4]}")
          print(f"TPs {TPs}, FPs {FPs}, FNs {FNs}, TNs {TNs}")
          print(f"Precision {total_pre} Recall {total_rec} F2-score {total_fscore}")
          print("**************************")
          df_result[col_list].to_excel(f'{parent_dir}{f1.split("/")[4]}/{f1.split("/")[4]}_results.xlsx', index=None)
          df_final = df_final.append(df_result, ignore_index=True)
  TPs = df_final['TP'].sum()
  TNs = df_final['TN'].sum()
  FPs = df_final['FP'].sum()
  FNs = df_final['FN'].sum()
  total_pre = 0 if (TPs == 0) & (FPs == 0) else TPs/(TPs + FPs)
  total_rec = 0 if (TPs == 0) & (FNs == 0) else TPs/(TPs + FNs)
  total_fscore = 0 if (total_pre == 0) & (total_rec == 0) else (5*total_pre*total_rec)/(4*total_pre + total_rec)
  print(f"Overall Results")
  print(f"TPs {TPs}, FPs {FPs}, FNs {FNs}, TNs {TNs}")
  print(f"Precision {total_pre} Recall {total_rec} F2-score {total_fscore}")
  # df_final[col_list].to_excel(f'{parent_dir}/all_results.xlsx', index=None)
  compute_single_table_for_all_DPAs(df_final, parent_dir)



def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, df_train, model_type):

  """Train the model for an epcoh and returns the training results for each epoch"""

  model = model.train()
  losses = []
  correct_predictions = 0
  predictions = []
  real_values = []

  for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
      predictions.extend(preds)
      real_values.extend(targets)

  predictions = torch.stack(predictions).cpu()
  real_values = torch.stack(real_values).cpu()
  # labels = np.unique(targets.cpu())
  labels = df_train.label.unique()
  result = classification_report(real_values, predictions, target_names=labels, labels=labels, output_dict=True)
  df_result = pd.DataFrame(result)
  # print(df_result.T)
  precision, recall, f1score, f2score = extract_results_from_report(df_result.T, model_type)

  return correct_predictions.double() / len(df_train), np.mean(losses), precision, recall, f1score, f2score


def eval_model(model, data_loader, loss_fn, device, df_val, model_type):

  """Validates the model during train for one epoch and returns the validation results"""

  model = model.eval()
  losses = []
  correct_predictions = 0
  predictions = []
  real_values = []
  with torch.no_grad():
      for d in data_loader:
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          targets = d["targets"].to(device)

          outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask
          )
          _, preds = torch.max(outputs, dim=1)
          # print(outputs)
          loss = loss_fn(outputs, targets)
          correct_predictions += torch.sum(preds == targets)
          losses.append(loss.item())
          predictions.extend(preds)
          real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  real_values = torch.stack(real_values).cpu()
  # labels = np.unique(targets.cpu())
  labels = df_val.label.unique()
  result = classification_report(real_values, predictions, target_names=labels, labels=labels, output_dict=True)
  df_result = pd.DataFrame(result)
  print(df_result.T)
  precision, recall, f1score, f2score = extract_results_from_report(df_result.T, model_type)
  return correct_predictions.double() / len(df_val), np.mean(losses), precision, recall, f1score, f2score


def start_training(model, model_name, loss_fn, optimizer, scheduler, train_data_loader, val_data_loader, EPOCHS,
                  BATCH_SIZE, lr, parent_dir, df_train, df_val, model_type):
  """This method train and validate a model for the given number of epochs, saves the best model and saves the model history in a dataframe."""

  history = []
  best_f_score = 0
  min_difference = 1
  bad_model_counter = 0
  parent_dir_1 = f"{parent_dir}/saved_models/"
  directory = f"Epochs_{EPOCHS}_BatchSize_{BATCH_SIZE}_{lr}"
  create_directory(directory, parent_dir_1)
  start = time.time()
  early_stopper = EarlyStopper(patience=5, min_delta=0)
  for epoch in range(EPOCHS):
      is_model_saved = 'No'
      print(f'Epoch {epoch + 1}/{EPOCHS}')
      print('-' * 10)
      train_acc, train_loss, t_precision, t_recall, t_f1score, t_f2score = train_epoch(
          model, train_data_loader, loss_fn, optimizer, device, scheduler, df_train, model_type)
      print(f'Train loss {train_loss} accuracy {train_acc}')
      print(f'Train precision {t_precision} recall {t_recall} F2-Score {t_f2score}')
      val_acc, val_loss, v_precision, v_recall, v_f1score, v_f2score = eval_model(
          model, val_data_loader, loss_fn, device, df_val, model_type)
      print(f'Val loss {val_loss} accuracy {val_acc}')
      print(f'Val precision {v_precision} recall {v_recall} F2-Score {v_f2score}')
      print()
      if v_f2score <= t_f2score:
          if early_stopper.early_stop_f_score(v_f1score):
              print(f'Early stop after {epoch + 1} epochs')
              break
      f2score_diff = t_f2score - v_f2score
      if (v_f2score > best_f_score) and (v_f2score <= t_f2score):
          # if f2score_diff < min_difference:
          torch.save(model.state_dict(), f'{parent_dir_1}/{directory}/{model_name}_state.bin')
          min_difference = f2score_diff
          best_f_score = v_f2score
          is_model_saved = 'Yes'
          print(f'Model saved for epoch {epoch + 1},  validation F2-score {v_f2score} difference is {f2score_diff}')
      else:
          bad_model_counter += 1
          if bad_model_counter >= EPOCHS:
              torch.save(model.state_dict(), f'{parent_dir_1}/{directory}/{model_name}_state.bin')
              is_model_saved = 'Yes'
              print(
                  f'Model saved for epoch {epoch + 1},  validation F2-score {v_f2score} difference is {f2score_diff}')

      history.append({"model": model_name, "train_acc": train_acc.item(), "train_loss": train_loss,
                      "val_acc": val_acc.item(), "val_loss": val_loss,
                      "train_precision": t_precision, "train_recall": t_recall, "train_f1_score": t_f1score,
                      "train_f2_score": t_f2score,
                      "val_precision": v_precision, "val_recall": v_recall, "val_f1_score": v_f1score,
                      "val_f2_score": v_f2score, "model_saved": is_model_saved
                      })
  end = time.time()
  history = pd.DataFrame(history)
  history['Time'] = (end - start) / 60
  return history


def get_predictions(model, test_set, labels_name, labels, data_loader, model_name, parent_dir,
                  directory, model_type='binary'):

  """This method is used to prediction results of a model both during training or testing. The result are saved as excel files."""

  model = model.eval()
  #print(parent_dir)
  create_directory('model_results', parent_dir)
  parent_dir_1 = f"{parent_dir}/model_results/"
  #print(parent_dir_1)
  create_directory(directory, parent_dir_1)
  df_output = pd.DataFrame()
  sentence_texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  correct_predictions = 0
  with torch.no_grad():
      for d in data_loader:
          texts = d["sentence_text"]
          input_ids = d["input_ids"].to(device)
          attention_mask = d["attention_mask"].to(device)
          targets = d["targets"].to(device)

          outputs = model(input_ids=input_ids, attention_mask=attention_mask)
          _, preds = torch.max(outputs, dim=1)
          correct_predictions += torch.sum(preds == targets)
          probs = F.softmax(outputs, dim=1)
          sentence_texts.extend(texts)
          predictions.extend(preds)
          prediction_probs.extend(probs)
          real_values.extend(targets)
  accuracy = correct_predictions.double() / len(test_set)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  df_output['sentence'] = sentence_texts
  df_output['prediction'] = predictions
  df_output['target'] = real_values
  # labels = test.target.unique().tolist()
  print(accuracy.item())
  if model_type != 'binary':
      labels_name.sort()
      labels.sort()
  result = classification_report(real_values, predictions, target_names=labels_name, labels=labels, output_dict=True)
  df_result = pd.DataFrame(result)
  df_result.T.to_excel(f'{parent_dir_1}/{directory}/{model_name}_pre_recall.xlsx')
  cm = confusion_matrix(real_values, predictions, labels=labels)
  df_cm = pd.DataFrame(cm, index=labels_name, columns=labels_name)
  df_cm.to_excel(f'{parent_dir_1}/{directory}/{model_name}_confusion_matrix.xlsx')
  df_output.to_excel(f'{parent_dir_1}/{directory}/{model_name}_output.xlsx')



def train_four_models(df_train, df_val, EPOCHS, BATCH_SIZE, lr, parent_dir, model_path, model_type='binary', cw=False):

  """This method itterates through the language models and train one by one"""

  df_history = pd.DataFrame()
  for key, model_name in PRE_TRAINED_MODELS.items():
      # df_history = pd.DataFrame()
      if key == 'BERT':
          print(f"Now processing model: {key}")
          tokenizer = BertTokenizer.from_pretrained(model_name, return_dict=False)
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, cw=cw)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          model = Bert_Classifier(len(df_train.target.unique()), model_name)
          model = model.to(device)
      elif key == 'RoBERTa':
          print(f"Now processing model: {key}")
          tokenizer = RobertaTokenizer.from_pretrained(model_name, return_dict=False)
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, cw=cw)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          model = Roberta_Classifier(len(df_train.target.unique()), model_name)
          model = model.to(device)
      elif key == 'ALBERT':
          print(f"Now processing model: {key}")
          tokenizer = AlbertTokenizer.from_pretrained(model_name, return_dict=False)
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, cw=cw)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          model = Albert_Classifier(len(df_train.target.unique()), model_name)
          model = model.to(device)
      else:
          print(f"Now processing model: {key}")
          tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=False)
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, cw=cw)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          model = LBert_Classifier(len(df_train.target.unique()), model_name)
          model = model.to(device)
      optimizer = AdamW(model.parameters(), lr=lr, correct_bias=False)
      total_steps = len(train_data_loader) * EPOCHS
      scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
      loss_fn = nn.CrossEntropyLoss().to(device)
      start = time.time()
      history = start_training(model, key, loss_fn, optimizer, scheduler, train_data_loader, val_data_loader, EPOCHS,
                                BATCH_SIZE, lr, model_path, df_train, df_val, model_type)
      end = time.time()
      df_history = pd.concat([df_history, history], ignore_index=True)
      df_history['EPOCHS'] = EPOCHS
      df_history['Batch_Size'] = BATCH_SIZE

      print(f"Processing time in minutes {(end - start) / 60}")

  df_history.to_excel(f"{parent_dir}/model_history/{key}_history_{EPOCHS}_{BATCH_SIZE}_{lr}.xlsx")



def start_experiments_binary_original_dataset(df, parent_dir_path, model_path, parent_dir):

  """This method runs the experiments to train all the binary models on the original dataset"""

  for req_id in reqs_list:
      train, test = load_dataset(df)
      size_req_id = train.loc[train.target == 'R16'].shape[0]
      # To undersample the dataset
      # train_sampled = under_sample_for_binary_classification(train, req_id)
      train_sampled = get_train_dataset_each_req(train, req_id)
      df_train, df_val = train_test_split(train_sampled, test_size=0.2, random_state=RANDOM_SEED,
                                          stratify=train_sampled['label'])
      print(f"Training dataset size: {df_train.shape}, Validation dataset size: {df_val.shape}")
      # To oversample the dataset
      # df_train = get_over_sampled_dataset_imblearn(df_train, df_train.target)
      dir_name = req_id
      create_directory(parent_dir, parent_dir_path)
      create_directory(parent_dir, model_path)
      create_directory(dir_name, parent_dir_path + parent_dir)
      create_directory(dir_name, model_path + parent_dir)
      create_directory('saved_models', model_path + parent_dir + '/' + dir_name + '/')
      create_directory('model_history', parent_dir_path + parent_dir + '/' + dir_name + '/')
      # create_directory('model_results', parent_dir_path + parent_dir + '/' + dir_name + '/')
      for lr in LEARNING_RATE_LIST:
          for EPOCH in EPOCHS_LIST:
              for BATCH_SIZE in BATCH_SIZE_LIST:
                  print(f'Epoch {EPOCH} and batch size {BATCH_SIZE}')
                  train_four_models(df_train, df_val, EPOCH, BATCH_SIZE, lr,
                                    parent_dir_path + parent_dir + '/' + dir_name,
                                    model_path + parent_dir + '/' + dir_name, 'binary')



def start_experiments_MCC_original_dataset(df, parent_dir, dir_name, model_path):

  """This method run the experiments to train all the multi-class classification models on the original dataset"""

  train, test = load_dataset(df)
  size_req_id = train.loc[train.target == 'R16'].shape[0]
  # To undersample the dataset
  # train = under_sample_other_label(train, size_req_id)
  df_train, df_val = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED, stratify=train['label'])
  print(f"Training dataset size: {df_train.shape}, Validation dataset size: {df_val.shape}")
  # To oversample the dataset
  # df_train = get_over_sampled_dataset_imblearn(df_train, df_train['target'])
  create_directory(dir_name, parent_dir)
  create_directory(dir_name, model_path)
  create_directory('saved_models', model_path + dir_name + '/')
  create_directory('model_history', parent_dir + dir_name + '/')
  # create_directory('model_results', parent_dir + dir_name + '/')
  for lr in LEARNING_RATE_LIST:
      for EPOCH in EPOCHS_LIST:
          for BATCH_SIZE in BATCH_SIZE_LIST:
              print(f'Epoch {EPOCH} and batch size {BATCH_SIZE}')
              train_four_models(df_train, df_val, EPOCH, BATCH_SIZE, lr, parent_dir + dir_name,
                                model_path + dir_name, 'mcc')



def test_binary_models_per_DPA(test, model_path, results_path):

    """This method tests all the binary models one by one and print the results"""

    model_name = "bert-base-uncased"
    df_final = pd.DataFrame()
    tokenizer = BertTokenizer.from_pretrained(model_name, return_dict=False)
    model = Bert_Classifier(2, model_name)
    for dpa in test.DPA.unique():
        df_result = pd.DataFrame()
        dpa_size = 0
        for req_id in reqs_list:
            df_dpa = test.loc[test.DPA == dpa]
            dpa_size = len(df_dpa)
            df_dpa = get_test_dataset_each_req(df_dpa, req_id)
            labels_name = ['other', req_id]
            labels = [0, 1]
            test_data_loader = create_data_loader(df_dpa, tokenizer, MAX_LEN, 64, cw=False)
            model.load_state_dict(torch.load(f'{model_path}{req_id}/BERT_state.bin', map_location=torch.device('cpu')))
            model = model.to(device)
            df_req_result = get_predictions_binary_test(model, df_dpa, labels_name, labels, test_data_loader, results_path, device, 'binary')
            df_req_result['Requirement'] = req_id
            df_result = df_result.append(df_req_result, ignore_index=True)
        col_list = ['DPA', 'Requirement', 'TP', 'FP', 'FN', 'TN']
        df_result['DPA'] = dpa
        if len(df_result) > 0:
            df_result.loc[df_result.TP > 0, ['TP', 'FP', 'FN', 'TN']] = 1.0, 0.0, 0.0, 0.0
            df_result.loc[(df_result.FN > 0) & (df_result.FP > 0), ['TP', 'FN', 'TN', 'FP']] = 1.0, 0.0, 0.0, 0.0
            df_result.loc[(df_result.FN > 0) & (df_result.FP == 0), ['TP', 'FN', 'TN', 'FP']] = 0.0, 1.0, 0.0, 0.0
            df_result.loc[(df_result.FN == 0) & (df_result.FP > 0), ['TP', 'FN', 'TN', 'FP']] = 0.0, 0.0, 0.0, 1.0
            df_result.loc[(df_result.FN == 0) & (df_result.FP == 0) & (df_result.TP == 0), 'TN'] = 1.0
            df_final = df_final.append(df_result, ignore_index=True)
            TPs = df_result['TP'].sum()
            TNs = df_result['TN'].sum()
            FPs = df_result['FP'].sum()
            FNs = df_result['FN'].sum()
            total_pre = 0 if (TPs == 0) & (FPs == 0) else TPs/(TPs + FPs)
            total_rec = 0 if (TPs == 0) & (FNs == 0) else TPs/(TPs + FNs)
            total_fscore = 0 if (total_pre == 0) & (total_rec == 0) else (5*total_pre*total_rec)/(4*total_pre + total_rec)
            print(f"Results of DPA {dpa}")
            print(f"TPs {TPs}, FPs {FPs}, FNs {FNs}, TNs {TNs}")
            print(f"Precision {total_pre} Recall {total_rec} F2-score {total_fscore}")
            df_result.sort_values(by=['Requirement'], ascending=True, inplace=True)
            df_result[col_list].to_excel(f'{results_path}/{dpa}_results.xlsx', index=None)

    compute_single_table_for_all_DPAs(df_final, results_path)

def test_MCC_models_per_DPA(test, results_path, model_path):
    """This method test the Multi-class classifition model and print the results"""

    model_name = "roberta-base"
    # labels_name = test.target.unique().tolist()
    # labels = test.label.unique()
    labels_name = label_names_list
    labels = labels_list
    tokenizer = RobertaTokenizer.from_pretrained(model_name, return_dict=False)
    model = Roberta_Classifier(20, model_name)
    model.load_state_dict(torch.load(f'{model_path}RoBERTa_state.bin', map_location=torch.device('cpu')))
    model = model.to(device)
    create_directory('DPA_results', results_path)
    for dpa in test.DPA.unique():
        df_dpa = test.loc[test.DPA == dpa]
        create_directory(dpa, results_path + 'DPA_results/')
        results_path_1 = f"{results_path}DPA_results/{dpa}/"
        test_data_loader = create_data_loader(df_dpa, tokenizer, MAX_LEN, 64, cw=False)
        get_predictions_test(model, df_dpa, labels_name, labels, test_data_loader, results_path_1, device, 'mcc')

    results_dir_path = f"{results_path}DPA_results/"
    compute_result_per_instance_per_DPA_MCC(results_dir_path)



def start_experiments_binary_ML(df, parent_dir, parent_dir_path, model_path):
  """This method runs the experiments to train all the binary models on the original dataset"""
  results_list = []
  for req_id in reqs_list:
      start = time.time()
      train, test = load_dataset(df)
      # train = under_sample_for_binary_classification(train, req_id)
      train = get_train_dataset_each_req(train, req_id)
      # _, train = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED, stratify=train['label'])
      ## For over sampling
      # train = get_over_sampled_dataset_imblearn(train, train.target)
      # train = train.reset_index()
      train_embeddings = extract_embeddings_sbert(train)
      end = time.time()
      dir_name = req_id
      create_directory(parent_dir, parent_dir_path)
      create_directory(parent_dir, model_path)
      create_directory(dir_name, model_path + '/' + parent_dir + '/')
      create_directory('saved_models', model_path + '/' + parent_dir + '/' + dir_name + '/')
      models = get_models()
      for key in models:
          print(f"Training model {key}")
          result = train_and_evaluate_model(train_embeddings['Embeddings'], train_embeddings['label'], models[key],
                                          key, model_path + '/' + parent_dir + '/' + dir_name + '/saved_models/')

          result['Requirement'] = req_id
          result['FE_time'] = (end - start) / 60
          results_list.append(result)
  df_results = pd.DataFrame.from_dict(results_list)
  df_results.to_excel(f'{parent_dir_path}/{parent_dir}/train_results.xlsx', index=None)


def get_models():
  """This method returns the machine learning models"""
  return {
      'SVM': SVC(),
      # MCC best params: c=10, gamma=0.1, kernel=rbf | binary best params:
      'RF': RandomForestClassifier(),
      # Default params performed well
      'LR': LogisticRegression()
      # Default params performed well
  }


def train_and_evaluate_model(X, Y, model, model_name, model_path):
  """This method trains and evaluates the binary machine learning models"""
  X_train, X_val, y_train, y_val = train_test_split(X.apply(pd.Series), Y, test_size=0.2,
                                                    random_state=RANDOM_SEED, stratify=Y)
  start = time.time()
  model.fit(X_train, y_train)
  # make predictions for test data
  y_pred = model.predict(X_val)
  predictions = [round(value) for value in y_pred]
  end = time.time()
  # evaluate predictions
  accuracy = accuracy_score(y_val, predictions)
  precision = precision_score(y_val, predictions)
  recall = recall_score(y_val, predictions)
  f1score = f1_score(y_val, predictions)
  f2score = (5 * precision * recall)/(4 * precision + recall) * 100.0
  print(f"Precision {precision * 100}")
  print(f"Recall {recall * 100}")
  print("Accuracy: %.2f%%" % (accuracy * 100.0))
  print(f"F1 Score: {f1score*100.0}")
  print(f"F2-Score: {f2score}")
  result_dict = {'Model': model_name, 'Precision': precision * 100.0,
                'Recall': recall * 100.0, 'F2-score': f2score, 'F1_Score': f1score * 100.0,
                'Accuracy': accuracy * 100.0, 'Training time': (end - start) / 60}
  # Save the model
  with open(f'{model_path}/{model_name}.pickle.dat', 'wb') as f:
      pickle.dump(model, f)

  return result_dict


def start_experiments_MCC_ML(df, parent_dir, dir_name, model_path):
  """This method runs the experiments to train all the multi-class classification models on the original dataset"""
  results_list = []
  start = time.time()
  train, test = load_dataset(df)
  # To undersample the train dataset
  # train = under_sample_other_label(train)
  df_train, df_val = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED, stratify=train['label'])
  # To oversample the train dataset
  # df_train = get_over_sampled_dataset_imblearn(df_train, df_train['target'])
  train_embeddings = extract_embeddings_sbert(train)
  end = time.time()
  create_directory(dir_name, parent_dir)
  create_directory(dir_name, model_path)
  create_directory('saved_models', model_path + dir_name + '/')
  create_directory('model_history', parent_dir + dir_name + '/')
  label_names = train.target.unique().tolist()
  labels = train.label.unique()
  models = get_models()
  for key in models:
      print(f"Training model {key}")
      result = train_and_evaluate_mcc_model(train_embeddings['Embeddings'], train_embeddings['label'], 
                                            models[key], key, label_names, labels,
                                  model_path + '/' + dir_name + '/saved_models/')
      result['FE_time'] = (end - start) / 60
      results_list.append(result)
  df_results = pd.DataFrame.from_dict(results_list)
  df_results.to_excel(f'{parent_dir}/{dir_name}/train_results.xlsx', index=None)


def train_and_evaluate_mcc_model(X, Y, model, model_name, label_names, labels, model_path):
  """This method trains and evaluates the multi-class classification machine learning models"""
  X_train, X_val, y_train, y_val = train_test_split(X.apply(pd.Series), Y, test_size=0.2,
                                                    random_state=RANDOM_SEED, stratify=Y)
  start = time.time()
  model.fit(X_train, y_train)
  # make predictions for test data
  y_pred = model.predict(X_val)
  predictions = [round(value) for value in y_pred]
  end = time.time()
  # evaluate predictions
  result = classification_report(y_val.tolist(), predictions, target_names=label_names, labels=labels,
                                output_dict=True)
  df_result = pd.DataFrame(result)
  print(df_result.T)
  result_dict = {'Model': model_name, 'Training time': (end - start) / 60}
  # Save the model
  with open(f'{model_path}/{model_name}.pickle.dat', 'wb') as f:
      pickle.dump(model, f)

  return result_dict



def train_MLP_model(df_train, df_val, model_name, EPOCH, BATCH_SIZE, lr, results_path, model_path, model_type):
    """This method trains and evaluates the binary machine learning models"""
    num_classes = len(df_train.label.unique())
    if model_name == 'NN':
        model = NeuralNet(input_size, hidden_size, num_classes)
    else:
        model = BiLSTM(input_size, hidden_size, hidden_size, num_classes, 2, True, 0.5)
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_data = DPA_Dataset(df_train)
    val_data = DPA_Dataset(df_val)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    start_training_DL(model, train_loader, val_loader, criterion, optimizer, df_train, df_val, model_type, results_path,
                  model_path, EPOCH, BATCH_SIZE, lr, model_name)


def start_experiments_binary_DL(df, model_name, parent_dir, parent_dir_path, model_path):
    """This method runs the experiments to train binary deep learning models (MLP and BiLSTM) on the original dataset"""
    if model_name == 'NN':
        EPOCH = 3 
        BATCH_SIZE = 32
        lr = 5e-5
    else:
        EPOCH = 3 
        BATCH_SIZE = 32
        lr = 0.0001

    df_time = pd.DataFrame()
    for req_id in reqs_list:
        start = time.time()
        train, test = load_dataset(df)
        # To undersample the train dataset
        # train = under_sample_for_binary_classification(train, req_id)
        train = get_train_dataset_each_req(train, req_id)
        df_train, df_val = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED,
                                          stratify=train['label'])
        df_train = df_train.reset_index()
        df_val = df_val.reset_index()
        # To oversample the train dataset
        # train = get_over_sampled_dataset_imblearn(train, train.target)
        train_embeddings = extract_embeddings_sbert(df_train)
        val_embeddings = extract_embeddings_sbert(df_val)
        end = time.time()
        df_time = df_time.append({'Requirement': req_id, 'FE_time': (end - start) / 60}, ignore_index=True)
        dir_name = req_id
        create_directory(parent_dir, parent_dir_path)
        create_directory(parent_dir, model_path)
        create_directory(dir_name, parent_dir_path + parent_dir)
        create_directory(dir_name, model_path + parent_dir)
        create_directory('saved_models', model_path + parent_dir + '/' + dir_name + '/')
        create_directory('model_history', parent_dir_path + parent_dir + '/' + dir_name + '/')
        # create_directory('model_results', parent_dir_path + parent_dir + '/' + dir_name + '/')
        # for lr in LEARNING_RATE_LIST:
        #     for EPOCH in EPOCHS_LIST:
        #         for BATCH_SIZE in BATCH_SIZE_LIST:
        print(f'Epoch {EPOCH} and batch size {BATCH_SIZE}')
        train_MLP_model(train_embeddings, val_embeddings, model_name, EPOCH, BATCH_SIZE, lr,
            parent_dir_path + parent_dir + '/' + dir_name,
            model_path + parent_dir + '/' + dir_name, 'binary')
    df_time.to_excel(f'{parent_dir_path}/FE_execution_time.xlsx')


def train_epochs_DL(model, train_loader, criterion, optimizer, df_train, device, model_name, BATCH_SIZE, model_type):
    """Train the model for an epcoh and return results for each epoch"""
    model = model.train()
    train_losses = []
    predictions = []
    real_values = []
    correct_predictions = 0
    # n_total_steps = len(train_loader)
    for i, (text_features, targets) in enumerate(train_loader):
        # Forward pass
        if model_name == 'NN':
            outputs = model(text_features)
        else:
            outputs = model(text_features.reshape(-1, 1, input_size))
        loss = criterion(outputs, targets.reshape(-1).type(torch.LongTensor))
        _, preds = torch.max(outputs, 1)
        correct_predictions += (preds == targets).sum().item()
        train_losses.append(loss.item())
        predictions.extend(preds)
        real_values.extend(targets.reshape(-1).type(torch.LongTensor))
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    labels = df_train.label.unique()
    result = classification_report(real_values, predictions, target_names=labels, labels=labels, output_dict=True)
    df_result = pd.DataFrame(result)
    #     print(df_result.T)
    precision, recall, f1score, f2score = extract_results_from_report(df_result.T, model_type)
    return correct_predictions / len(df_train), np.mean(train_losses), precision, recall, f1score, f2score


def eval_epochs_DL(model, val_loader, criterion, df_val, device, model_name, BATCH_SIZE, model_type):
    """Validates the model during train for one epoch and returns the validation results"""
    # Test the model
    model = model.eval()
    val_losses = []
    predictions = []
    real_values = []
    correct_predictions = 0
    with torch.no_grad():
        for text_features, targets in val_loader:
            text_features = text_features
            targets = targets
            if model_name == 'NN':
                outputs = model(text_features)
            else:
                outputs = model(text_features.reshape(-1, 1, input_size))
            # max returns (value ,index)
            _, preds = torch.max(outputs, 1)
            correct_predictions += (preds == targets).sum().item()
            loss = criterion(outputs, targets.reshape(-1).type(torch.LongTensor))

            val_losses.append(loss.item())
            predictions.extend(preds)
            real_values.extend(targets.reshape(-1).type(torch.LongTensor))

    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    labels = df_val.label.unique()
    result = classification_report(real_values, predictions, target_names=labels, labels=labels, output_dict=True)
    df_result = pd.DataFrame(result)
    precision, recall, f1score, f2score = extract_results_from_report(df_result.T, model_type)
    return correct_predictions / len(df_val), np.mean(val_losses), precision, recall, f1score, f2score



def start_experiments_MCC_DL(df, model_name, parent_dir, dir_name, model_path):
    """This method runs the experiments to train multi-class classifiction deep learning models (MLP and BiLSTM) on the original dataset"""
    if model_name == 'NN':
        EPOCH = 3 
        BATCH_SIZE = 32
        lr = 5e-5
    else:
        EPOCH = 3 
        BATCH_SIZE = 32
        lr = 0.0001
    df_time = pd.DataFrame()
    start = time.time()
    train, test = load_dataset(df)
    size_req_id = train.loc[train.target == 'R16'].shape[0]
    # To undersample the train dataset
    # train = under_sample_other_label(train, size_req_id)
    df_train, df_val = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED, stratify=train['label'])
    print(f"Training dataset size: {df_train.shape}, Validation dataset size: {df_val.shape}")
    # To oversample the train dataset
    # df_train = get_over_sampled_dataset_imblearn(df_train, df_train['target'])
    print(df_train.target.value_counts())
    print(df_train.label.value_counts())
    print(f'Size of the dataset {df_train.shape}')
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    train_embeddings = extract_embeddings_sbert(df_train)
    val_embeddings = extract_embeddings_sbert(df_val)
    end = time.time()
    df_time = df_time.append({'FE_time': (end - start) / 60}, ignore_index=True)
    create_directory(dir_name, parent_dir)
    create_directory(dir_name, model_path)
    create_directory('saved_models', model_path + dir_name + '/')
    create_directory('model_history', parent_dir + dir_name + '/')
    create_directory('model_results', parent_dir + dir_name + '/')
    train_MLP_model(train_embeddings, val_embeddings, model_name, EPOCH, BATCH_SIZE, lr, parent_dir + dir_name,
                          model_path + dir_name, 'mcc')
    df_time.to_excel(f'{parent_dir}/{dir_name}/FE_execution_time.xlsx')



def start_training_DL(model, train_loader, val_loader, criterion, optimizer, df_train, df_val, model_type,
                  results_path, model_path, EPOCHS, BATCH_SIZE, LR, model_name):
    """This method train and validate a model for the given number of epochs, saves the best model and saves the model history in a dataframe."""
    # Train the model
    history = []
    best_f_score = 0
    min_difference = 1
    bad_model_counter = 0
    start = time.time()
    early_stopper = EarlyStopper(patience=5, min_delta=0)
    for epoch in range(EPOCHS):
        is_model_saved = 'No'
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        train_acc, train_loss, t_precision, t_recall, t_f1score, t_f2score = train_epochs_DL(model, train_loader,
                                                                                        criterion, optimizer,
                                                                                        df_train, device, model_name,
                                                                                        BATCH_SIZE, model_type)
        print(f'Train loss {train_loss} accuracy {train_acc}')
        print(f'Train precision {t_precision} recall {t_recall} F2-Score {t_f2score}')

        val_acc, val_loss, v_precision, v_recall, v_f1score, v_f2score = eval_epochs_DL(model, val_loader, criterion,
                                                                                    df_val, device, model_name,
                                                                                    BATCH_SIZE, model_type)
        print(f'Val loss {val_loss} accuracy {val_acc}')
        print(f'Val precision {v_precision} recall {v_recall} F2-Score {v_f2score}')
        print()

        if v_f2score <= t_f2score:
            if early_stopper.early_stop_f_score(v_f1score):
                print(f'Early stop after {epoch + 1} epochs')
                break
        f2score_diff = t_f2score - v_f2score
        if (v_f2score > best_f_score) and (v_f2score <= t_f2score):
          # if f2score_diff < min_difference:
            torch.save(model.state_dict(),
                      f'{model_path}/saved_models/{model_name}_state_{BATCH_SIZE}_{EPOCHS}_{LR}.bin')
            min_difference = f2score_diff
            best_f_score = v_f2score
            is_model_saved = 'Yes'
            print(f'Model saved for epoch {epoch + 1},  validation F2-score {v_f2score} difference is {f2score_diff}')
        else:
            bad_model_counter += 1
            if bad_model_counter >= EPOCHS:
                torch.save(model.state_dict(),
                          f'{model_path}/saved_models/{model_name}_state_{BATCH_SIZE}_{EPOCHS}_{LR}.bin')
                is_model_saved = 'Yes'
                print(
                  f'Model saved for epoch {epoch + 1},  validation F2-score {v_f2score} difference is {f2score_diff}')
        history.append({"model": model_name, "train_acc": train_acc, "train_loss": train_loss,
                      "val_acc": val_acc, "val_loss": val_loss,
                      "train_precision": t_precision, "train_recall": t_recall, "train_f1_score": t_f1score,
                      "train_f2_score": t_f2score,
                      "val_precision": v_precision, "val_recall": v_recall, "val_f1_score": v_f1score,
                      "val_f2_score": v_f2score, "model_saved": is_model_saved
                      })
    end = time.time()
    history = pd.DataFrame(history)
    history['Time'] = (end - start) / 60
    history['EPOCHS'] = EPOCHS
    history['Batch_Size'] = BATCH_SIZE
    print(f"Processing time in minutes {(end - start) / 60}")
    history.to_excel(f"{results_path}/model_history/history_{EPOCHS}_{BATCH_SIZE}_{LR}.xlsx")


def extract_embeddings_sbert(data):
    """This method is used to extract embeddings for the Sentence-BERT model"""
    df_embeddings = pd.DataFrame()
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    model.max_seq_length = 160
    embeddings = model.encode(data['Sentence'], convert_to_tensor=True)
    for sent_index in range(len(data)):
        df_embeddings = pd.concat([df_embeddings, pd.DataFrame.from_records([{'ID': data['ID'][sent_index],
              'Sentence': data['Sentence'][sent_index],
              'Embeddings': embeddings[sent_index].tolist(),
              'label': data['label'][sent_index], 'target': data['target'][sent_index]}])], ignore_index=True)
    return df_embeddings



def few_shot_learning_MCC_model(train, size, parent_dir, dir_name, model_path):
    """This method trains and validates a multi-class classification FSL model"""
    df_time = pd.DataFrame()
    model = SetFitModel.from_pretrained(ST_model)
    model = model.to(device)
    # print(model.model_body)
    val, train = train_test_split(train, test_size=size, random_state=RANDOM_SEED, stratify=train['label'])
    train = Dataset.from_dict(train)
    val = Dataset.from_dict(val)
    trainer = SetFitTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=val,
        loss_class=CosineSimilarityLoss,
        num_iterations=20,
        num_epochs=1,
        batch_size=16,
        column_mapping={'Sentence': 'text', 'label': 'label'}
    )
    create_directory(dir_name, parent_dir)
    create_directory(dir_name, model_path)
    create_directory('saved_models', model_path + dir_name + '/')
    create_directory('model_results', parent_dir + dir_name + '/')
    start = time.time()
    trainer.train()
    end = time.time()
    print(f"Training time of the model {(end - start) / 60}")
    df_time = df_time.append({'Model': 'FSL', 'Dataset_Size': size, 'time': (end - start) / 60}, ignore_index=True)
    metric = trainer.evaluate()
    print(metric)
    model._save_pretrained(f"{model_path + dir_name + '/saved_models/'}fsl_model.bin")
    df_time.to_excel(f'{parent_dir}{dir_name}/training_time.xlsx')


def few_shot_learning_binary_model(df, size, parent_dir, dir_name, model_path):
  """This method trains and validates the binary class FSL models"""
  df_time = pd.DataFrame()
  model = SetFitModel.from_pretrained(ST_model)
  create_directory(dir_name, parent_dir)
  create_directory(dir_name, model_path)
  for req_id in reqs_list[:2]:
    train, test = load_dataset(df)
    df_val, df_train = train_test_split(train, test_size=size, random_state=RANDOM_SEED, stratify=train['label'])
    df_train = get_train_dataset_each_req(df_train, req_id)
    print(f"Training dataset size: {df_train.shape}, Validation dataset size: {df_val.shape}")
    df_train = Dataset.from_dict(df_train)
    df_val = Dataset.from_dict(df_val)
    trainer = SetFitTrainer(
      model=model,
      train_dataset=df_train,
      eval_dataset=df_val,
      loss_class=CosineSimilarityLoss,
      num_iterations=20,
      num_epochs=1,
      batch_size=16,
      column_mapping={'Sentence': 'text', 'label': 'label'}
    )
    create_directory(req_id, parent_dir + dir_name + '/')
    create_directory(req_id, model_path + dir_name + '/')
    parent_dir_1 = parent_dir + dir_name + '/' + req_id + '/'
    model_dir_2 = model_path + dir_name + '/' + req_id + '/'
    create_directory('saved_models', model_dir_2)
    create_directory('model_results', parent_dir_1)
    start = time.time()
    trainer.train()
    end = time.time()
    print(f"Training time of the model {(end - start) / 60}")
    df_time = df_time.append({'Model': 'FSL', 'Req_id': req_id, 'Dataset_Size': size,
                            'time': (end - start) / 60}, ignore_index=True)
    metric = trainer.evaluate()
    print(metric)
    model._save_pretrained(f"{model_dir_2}saved_models/fsl_model.bin")
  df_time.to_excel(f'{parent_dir}{dir_name}/training_time.xlsx')


