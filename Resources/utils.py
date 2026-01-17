import os
import numpy as np
import pandas as pd
import transformers
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, RobertaModel, \
    RobertaTokenizer, AlbertModel, AlbertTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM, \
    pipeline as hf_pipeline
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
    "Gemma": "google/gemma-3-1b-it",
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# OPP-115 Privacy Policy Dataset - 9 classes
OPP115_CLASSES = {
    'DataCollection': 0,
    'ThirdPartySharing': 1,
    'UserRights': 2,
    'DataRetention': 3,
    'DataSecurity': 4,
    'PolicyChange': 5,
    'DoNotTrack': 6,
    'SpecialAudiences': 7,
    'Other': 8
}
N_CLASSES = 9
label_names_list = list(OPP115_CLASSES.keys())
labels_list = list(OPP115_CLASSES.values())


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


class Gemma_Classifier(nn.Module):

  """This class defines the Gemma-3 model for classification using mean pooling"""

  def __init__(self, n_classes, model_name):
      super(Gemma_Classifier, self).__init__()
      self.gemma = AutoModel.from_pretrained(model_name, return_dict=True, trust_remote_code=True)
      self.drop = nn.Dropout(p=DROPOUT_PROBABILITY)
      self.out = nn.Linear(self.gemma.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):
      outputs = self.gemma(
          input_ids=input_ids,
          attention_mask=attention_mask
      )
      # Use mean pooling over last hidden state (Gemma doesn't have pooled output like BERT)
      last_hidden_state = outputs.last_hidden_state
      # Apply attention mask for proper mean pooling
      input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
      sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
      sum_mask = input_mask_expanded.sum(1)
      sum_mask = torch.clamp(sum_mask, min=1e-9)
      pooled_output = sum_embeddings / sum_mask
      output = self.drop(pooled_output)
      return self.out(output)


class PrivacyPolicyDataset(Dataset):
    """This class defines a Dataset object for OPP-115 privacy policy dataset"""
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
    """Load OPP-115 dataset - labels are already pre-encoded in the CSV"""
    train = df.loc[df['dataset_type'] == 'train']
    test = df.loc[df['dataset_type'] == 'test']
    print(f'Train dataset size: {train.shape[0]}, Test dataset size: {test.shape[0]}')
    print(f'Total labels in train dataset: {len(train.target.unique())}')
    print(f'Total labels in test dataset: {len(test.target.unique())}')
    print(f'Class distribution in train:\\n{train.target.value_counts()}')
    return train, test


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


def extract_results_from_report(report, model_type='mcc'):
    """Extract precision, recall, f1score, f2score from classification report for multi-class classification"""
    precision, recall, f1score, f2score = 0.0, 0.0, 0.0, 0.0

    # Multi-class classification - use macro average
    report = report.tail(2)
    precision = report.iloc[0][0]
    recall = report.iloc[0][1]
    f1score = report.iloc[0][2]
    if (4 * precision + recall) > 0:
        f2score = (5 * precision * recall) / (4 * precision + recall)
    else:
        f2score = 0.0

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


def get_predictions_test(model, test_set, labels_name, labels, data_loader, parent_dir, mydevice, model_type='mcc'):
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
    # Sort labels for consistent ordering in multi-class classification
    labels_name.sort()
    labels.sort()
    cm = confusion_matrix(real_values, predictions, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels_name, columns=labels_name)
    df_cm.to_excel(f'{parent_dir}/confusion_matrix.xlsx') 


def compute_single_table_for_all_policies(df_final, results_path):
    """Compute aggregated results for all privacy policies"""
    col_list = ['Category', 'TP', 'FP', 'FN', 'TN']
    df_results = pd.DataFrame()
    for cat in df_final.Category.unique():
        df_cat = df_final.loc[df_final.Category == cat]
        df_results = pd.concat([df_results, pd.DataFrame([{
            'Category': cat,
            'TP': df_cat.TP.sum(),
            'FP': df_cat.FP.sum(),
            'FN': df_cat.FN.sum(),
            'TN': df_cat.TN.sum()}])], ignore_index=True)
    df_results[col_list].to_excel(f'{results_path}/all_results.xlsx', index=None)


def combine_all_policies_results(parent_dir):
    """Combine results from all privacy policies"""
    df_final = pd.DataFrame()
    for dir_name in os.listdir(parent_dir):
        print(dir_name)
        policy = compute_result_per_instance_per_policy_MCC(parent_dir+dir_name)
        df_final = pd.concat([df_final, policy], axis=0)
    return df_final 


def compute_result_per_instance_per_policy_MCC(parent_dir):
  """Compute results per instance for multi-class classification on privacy policies"""
  df_final = pd.DataFrame()
  for root, subdirectories, files in os.walk(parent_dir):
    for subdirectory in subdirectories:
      f1 = os.path.join(root, subdirectory)
      for filename in os.listdir(os.path.join(root, subdirectory)):
        df_result = pd.DataFrame()
        f = os.path.join(os.path.join(root, subdirectory), filename)
        if os.path.isfile(f):
          if filename.endswith('xlsx') or filename.endswith('xls'):
            if filename.split('_')[0] == 'confusion':
              df = pd.read_excel(f)
              for cat in df['Unnamed: 0'].unique():
                df.loc[df[cat] > 1, cat] = 1
              for cat, i in zip(df['Unnamed: 0'].unique(), range(0, N_CLASSES)):
                row = df.loc[df['Unnamed: 0'] == cat]
                FNs = row.sum(axis=1)
                FPs = df[cat].sum()
                if df[cat][i] > 0:
                    TP, FP, TN, FN = 1.0, 0.0, 0.0, 0.0
                elif (FPs > 0) & (FNs.item() > 0):
                    TP, FP, TN, FN = 1.0, 0.0, 0.0, 0.0
                elif (FPs == 0) & (FNs.item() > 0):
                    TP, FP, TN, FN = 0.0, 0.0, 0.0, 1.0
                elif (FPs > 0) & (FNs.item() == 0):
                    TP, FP, TN, FN = 0.0, 1.0, 0.0, 0.0
                elif (FPs == 0) & (FNs.item() == 0):
                    TP, FP, TN, FN = 0.0, 0.0, 1.0, 0.0
                df_result = pd.concat([df_result, pd.DataFrame([{
                    'Policy': f1.split('/')[4],
                    'Category': cat,
                    'TN': TN, 'FP': FP, 'TP': TP,
                    'FN': FN }])], ignore_index=True)
        col_list = ['Policy', 'Category', 'TP', 'FP', 'FN', 'TN']
        if len(df_result) > 0:
          df_result.sort_values(by=['Category'], ascending=True, inplace=True)
          df_result = df_result.loc[df_result.Category.isin(label_names_list)]
          TPs = df_result['TP'].sum()
          TNs = df_result['TN'].sum()
          FPs = df_result['FP'].sum()
          FNs = df_result['FN'].sum()
          total_pre = 0 if (TPs == 0) & (FPs == 0) else TPs/(TPs + FPs)
          total_rec = 0 if (TPs == 0) & (FNs == 0) else TPs/(TPs + FNs)
          total_fscore = 0 if (total_pre == 0) & (total_rec == 0) else (5*total_pre*total_rec)/(4*total_pre + total_rec)
          print(f"Results of Policy {f1.split('/')[4]}")
          print(f"TPs {TPs}, FPs {FPs}, FNs {FNs}, TNs {TNs}")
          print(f"Precision {total_pre} Recall {total_rec} F2-score {total_fscore}")
          print("**************************")
          df_result[col_list].to_excel(f'{parent_dir}{f1.split("/")[4]}/{f1.split("/")[4]}_results.xlsx', index=None)
          df_final = pd.concat([df_final, df_result], ignore_index=True)
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
  compute_single_table_for_all_policies(df_final, parent_dir)



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
                  directory, model_type='mcc'):

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
  print(accuracy.item())
  # Sort labels for consistent ordering in multi-class classification
  labels_name.sort()
  labels.sort()
  result = classification_report(real_values, predictions, target_names=labels_name, labels=labels, output_dict=True)
  df_result = pd.DataFrame(result)
  df_result.T.to_excel(f'{parent_dir_1}/{directory}/{model_name}_pre_recall.xlsx')
  cm = confusion_matrix(real_values, predictions, labels=labels)
  df_cm = pd.DataFrame(cm, index=labels_name, columns=labels_name)
  df_cm.to_excel(f'{parent_dir_1}/{directory}/{model_name}_confusion_matrix.xlsx')
  df_output.to_excel(f'{parent_dir_1}/{directory}/{model_name}_output.xlsx')



def train_four_models(df_train, df_val, EPOCHS, BATCH_SIZE, lr, parent_dir, model_path, model_type='mcc', cw=True):

  """This method iterates through the language models and trains them one by one for multi-class classification"""

  df_history = pd.DataFrame()
  for key, model_name in PRE_TRAINED_MODELS.items():
      if key == 'BERT':
          print(f"Now processing model: {key}")
          tokenizer = BertTokenizer.from_pretrained(model_name, return_dict=False)
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, cw=cw)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          model = Bert_Classifier(N_CLASSES, model_name)
          model = model.to(device)
      elif key == 'RoBERTa':
          print(f"Now processing model: {key}")
          tokenizer = RobertaTokenizer.from_pretrained(model_name, return_dict=False)
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, cw=cw)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          model = Roberta_Classifier(N_CLASSES, model_name)
          model = model.to(device)
      elif key == 'Gemma':
          print(f"Now processing model: {key}")
          tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
          if tokenizer.pad_token is None:
              tokenizer.pad_token = tokenizer.eos_token
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, cw=cw)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          model = Gemma_Classifier(N_CLASSES, model_name)
          model = model.to(device)
      elif key == 'ALBERT':
          print(f"Now processing model: {key}")
          tokenizer = AlbertTokenizer.from_pretrained(model_name, return_dict=False)
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, cw=cw)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          model = Albert_Classifier(N_CLASSES, model_name)
          model = model.to(device)
      else:
          print(f"Now processing model: {key}")
          tokenizer = AutoTokenizer.from_pretrained(model_name, return_dict=False)
          train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE, cw=cw)
          val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
          model = LBert_Classifier(N_CLASSES, model_name)
          model = model.to(device)
      optimizer = optim.AdamW(model.parameters(), lr=lr, correct_bias=False)
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



def start_experiments_MCC_original_dataset(df, parent_dir, dir_name, model_path):

  """This method runs the experiments to train all the multi-class classification models on the OPP-115 dataset"""

  train, test = load_dataset(df)
  df_train, df_val = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED, stratify=train['label'])
  print(f"Training dataset size: {df_train.shape}, Validation dataset size: {df_val.shape}")
  create_directory(dir_name, parent_dir)
  create_directory(dir_name, model_path)
  create_directory('saved_models', model_path + dir_name + '/')
  create_directory('model_history', parent_dir + dir_name + '/')
  for lr in LEARNING_RATE_LIST:
      for EPOCH in EPOCHS_LIST:
          for BATCH_SIZE in BATCH_SIZE_LIST:
              print(f'Epoch {EPOCH} and batch size {BATCH_SIZE}')
              train_four_models(df_train, df_val, EPOCH, BATCH_SIZE, lr, parent_dir + dir_name,
                                model_path + dir_name, 'mcc', cw=True)



def test_MCC_models_per_policy(test, results_path, model_path):
    """This method tests the Multi-class classification model on privacy policies and prints the results"""

    model_name = "roberta-base"
    labels_name = label_names_list
    labels = labels_list
    tokenizer = RobertaTokenizer.from_pretrained(model_name, return_dict=False)
    model = Roberta_Classifier(N_CLASSES, model_name)
    model.load_state_dict(torch.load(f'{model_path}RoBERTa_state.bin', map_location=torch.device('cpu')))
    model = model.to(device)
    create_directory('policy_results', results_path)
    for policy in test.policy_name.unique():
        df_policy = test.loc[test.policy_name == policy]
        create_directory(policy, results_path + 'policy_results/')
        results_path_1 = f"{results_path}policy_results/{policy}/"
        test_data_loader = create_data_loader(df_policy, tokenizer, MAX_LEN, 64, cw=False)
        get_predictions_test(model, df_policy, labels_name, labels, test_data_loader, results_path_1, device, 'mcc')

    results_dir_path = f"{results_path}policy_results/"
    compute_result_per_instance_per_policy_MCC(results_dir_path)



def get_models():
  """This method returns the machine learning models for multi-class classification"""
  return {
      'SVM': SVC(),
      'RF': RandomForestClassifier(),
      'LR': LogisticRegression(max_iter=1000)
  }


def start_experiments_MCC_ML(df, parent_dir, dir_name, model_path):
  """This method runs the experiments to train all the multi-class classification ML models on the OPP-115 dataset"""
  results_list = []
  start = time.time()
  train, test = load_dataset(df)
  df_train, df_val = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED, stratify=train['label'])
  train_embeddings = extract_embeddings_sbert(train)
  end = time.time()
  create_directory(dir_name, parent_dir)
  create_directory(dir_name, model_path)
  create_directory('saved_models', model_path + dir_name + '/')
  create_directory('model_history', parent_dir + dir_name + '/')
  models = get_models()
  for key in models:
      print(f"Training model {key}")
      result = train_and_evaluate_mcc_model(train_embeddings['Embeddings'], train_embeddings['label'], 
                                            models[key], key, label_names_list, labels_list,
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



def train_MLP_model(df_train, df_val, model_name, EPOCH, BATCH_SIZE, lr, results_path, model_path, model_type='mcc'):
    """This method trains and evaluates deep learning models (MLP and BiLSTM) for multi-class classification"""
    num_classes = N_CLASSES
    if model_name == 'NN':
        model = NeuralNet(input_size, hidden_size, num_classes)
    else:
        model = BiLSTM(input_size, hidden_size, hidden_size, num_classes, 2, True, 0.5)
    # Loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_data = PrivacyPolicyDataset(df_train)
    val_data = PrivacyPolicyDataset(df_val)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=True)
    start_training_DL(model, train_loader, val_loader, criterion, optimizer, df_train, df_val, model_type, results_path,
                  model_path, EPOCH, BATCH_SIZE, lr, model_name)


def train_epochs_DL(model, train_loader, criterion, optimizer, df_train, device, model_name, BATCH_SIZE, model_type='mcc'):
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


def eval_epochs_DL(model, val_loader, criterion, df_val, device, model_name, BATCH_SIZE, model_type='mcc'):
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
    """This method runs experiments to train multi-class classification deep learning models (MLP and BiLSTM) on OPP-115"""
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
    df_train, df_val = train_test_split(train, test_size=0.2, random_state=RANDOM_SEED, stratify=train['label'])
    print(f"Training dataset size: {df_train.shape}, Validation dataset size: {df_val.shape}")
    print(df_train.target.value_counts())
    print(df_train.label.value_counts())
    print(f'Size of the dataset {df_train.shape}')
    df_train = df_train.reset_index()
    df_val = df_val.reset_index()
    train_embeddings = extract_embeddings_sbert(df_train)
    val_embeddings = extract_embeddings_sbert(df_val)
    end = time.time()
    df_time = pd.concat([df_time, pd.DataFrame([{'FE_time': (end - start) / 60}])], ignore_index=True)
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
    """This method trains and validates a multi-class classification FSL model for OPP-115"""
    df_time = pd.DataFrame()
    model = SetFitModel.from_pretrained(ST_model)
    model = model.to(device)
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
    df_time = pd.concat([df_time, pd.DataFrame([{'Model': 'FSL', 'Dataset_Size': size, 'time': (end - start) / 60}])], ignore_index=True)
    metric = trainer.evaluate()
    print(metric)
    model._save_pretrained(f"{model_path + dir_name + '/saved_models/'}fsl_model.bin")
    df_time.to_excel(f'{parent_dir}{dir_name}/training_time.xlsx')


