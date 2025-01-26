import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

import torch
from torchmetrics import ConfusionMatrix, F1Score, AUROC

import time
import os


def binarize_label(df):
    df.rename(columns={"Label": "label_string"}, inplace=True)
    df['Label'] = np.where(df['label_string'] == "Benign", np.zeros(df['label_string'].shape, dtype='int'),
                           np.ones(df['label_string'].shape, dtype='int'))
    df.drop(columns=['label_string'], inplace=True)


# get dataset
dirpath = "drive/MyDrive/Dataset1/"
proc_cols = [
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Len",
    "Bwd Header Len",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Min",
    "Pkt Len Max",
    "Pkt Len Mean",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "CWE Flag Count",
    "ECE Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Fwd Seg Size Avg",
    "Bwd Seg Size Avg",
    "Fwd Byts/b Avg",
    "Fwd Pkts/b Avg",
    "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg",
    "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts",
    "Subflow Fwd Byts",
    "Subflow Bwd Pkts",
    "Subflow Bwd Byts",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
]
proc_types = {
    "Dst Port": "int64",
    "Protocol": "int64",
    "Timestamp": "string",
    "Flow Duration": "int64",
    "Tot Fwd Pkts": "int64",
    "Tot Bwd Pkts": "int64",
    "TotLen Fwd Pkts": "float64",
    "TotLen Bwd Pkts": "float64",
    "Fwd Pkt Len Max": "float64",
    "Fwd Pkt Len Min": "float64",
    "Fwd Pkt Len Mean": "float64",
    "Fwd Pkt Len Std": "float64",
    "Bwd Pkt Len Max": "float64",
    "Bwd Pkt Len Min": "float64",
    "Bwd Pkt Len Mean": "float64",
    "Bwd Pkt Len Std": "float64",
    "Flow Byts/s": "float64",
    "Flow Pkts/s": "float64",
    "Flow IAT Mean": "float64",
    "Flow IAT Std": "float64",
    "Flow IAT Max": "float64",
    "Flow IAT Min": "float64",
    "Fwd IAT Tot": "float64",
    "Fwd IAT Mean": "float64",
    "Fwd IAT Std": "float64",
    "Fwd IAT Max": "float64",
    "Fwd IAT Min": "float64",
    "Bwd IAT Tot": "float64",
    "Bwd IAT Mean": "float64",
    "Bwd IAT Std": "float64",
    "Bwd IAT Max": "float64",
    "Bwd IAT Min": "float64",
    "Fwd PSH Flags": "int64",
    "Bwd PSH Flags": "int64",
    "Fwd URG Flags": "int64",
    "Bwd URG Flags": "int64",
    "Fwd Header Len": "int64",
    "Bwd Header Len": "int64",
    "Fwd Pkts/s": "float64",
    "Bwd Pkts/s": "float64",
    "Pkt Len Min": "float64",
    "Pkt Len Max": "float64",
    "Pkt Len Mean": "float64",
    "Pkt Len Std": "float64",
    "Pkt Len Var": "float64",
    "FIN Flag Cnt": "int64",
    "SYN Flag Cnt": "int64",
    "RST Flag Cnt": "int64",
    "PSH Flag Cnt": "int64",
    "ACK Flag Cnt": "int64",
    "URG Flag Cnt": "int64",
    "CWE Flag Count": "int64",
    "ECE Flag Cnt": "int64",
    "Down/Up Ratio": "float64",
    "Pkt Size Avg": "float64",
    "Fwd Seg Size Avg": "float64",
    "Bwd Seg Size Avg": "float64",
    "Fwd Byts/b Avg": "int64",
    "Fwd Pkts/b Avg": "int64",
    "Fwd Blk Rate Avg": "int64",
    "Bwd Byts/b Avg": "int64",
    "Bwd Pkts/b Avg": "int64",
    "Bwd Blk Rate Avg": "int64",
    "Subflow Fwd Pkts": "int64",
    "Subflow Fwd Byts": "int64",
    "Subflow Bwd Pkts": "int64",
    "Subflow Bwd Byts": "int64",
    "Init Fwd Win Byts": "int64",
    "Init Bwd Win Byts": "int64",
    "Fwd Act Data Pkts": "int64",
    "Fwd Seg Size Min": "int64",
    "Active Mean": "float64",
    "Active Std": "float64",
    "Active Max": "float64",
    "Active Min": "float64",
    "Idle Mean": "float64",
    "Idle Std": "float64",
    "Idle Max": "float64",
    "Idle Min": "float64",
    "Label": "string",
}
files = os.listdir(dirpath)
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Loading dataset...")
frames = list()
for f in files:
    print(f"Loaded file {f}")
    partial_df = pd.read_csv(dirpath + f, sep=',', usecols=proc_cols)
    frames.append(partial_df)
print("Combining frames...")
df = pd.concat(frames)
del frames
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

###preprocessing

# binarize labels for benign/malicious traffic
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Processing Label column for binary classifier...")
binarize_label(df)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# convert timestamp to datetime and infer frequency
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Converting timestamp to datetime data...")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S")
#freq = pd.infer_freq(df['Timestamp'])
#del df_time
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# replace inf values and deal with NaN values by replacing them with 0
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Replacing infinite and NaN values...")
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")

# separate categorical and numerical features
NUMERICAL_FEATURES = df.select_dtypes(include="number").columns.tolist()
CATEGORICAL_FEATURES = df.select_dtypes(exclude="number").columns.tolist()
print(
    f"Numericals: {len(NUMERICAL_FEATURES)}; Categoricals: {len(CATEGORICAL_FEATURES)}"
)
CATEGORICAL_FEATURES.remove('Timestamp')
#CATEGORICAL_FEATURES.remove('Label')
targets_list = df['Label'].unique().tolist()

# split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
train, test = train_test_split(df, stratify=df["Label"], test_size=0.2, random_state=42)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")
num_classes = len(set(train["Label"].values.ravel()))

# prepare model
data_config = DataConfig(
    target=['Label'],  # target should always be a list.
    continuous_cols=NUMERICAL_FEATURES,
    categorical_cols=CATEGORICAL_FEATURES,
    date_columns=[('Timestamp','T',"%d/%m/%Y %H:%M:%S")],
    num_workers=15
)

trainer_config = TrainerConfig(
    #     auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=131072,
    max_epochs=1,
    early_stopping="valid_loss",  # Monitor valid_loss for early stopping
    early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
    early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
    checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
    load_best=True,  # After training, load the best checkpoint
)

optimizer_config = OptimizerConfig()

head_config = LinearHeadConfig(
    layers="",  # No additional layer in head, just a mapping layer to output_dim
    dropout=0.1,
    initialization="kaiming",
)._dict_  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

model_config = TabNetModelConfig(
    task="classification",
    learning_rate=1e-3,
    head="LinearHead",  # Linear Head
    head_config=head_config,  # Linear Head Config
    metrics=['accuracy', 'f1_score', 'precision', 'recall', 'auroc'], #found in Lib/site-packages/torchmetrics/functional/_init_.py
    metrics_prob_input=[False, False, False, False, True],
    metrics_params=[{'task': 'binary', 'num_classes': num_classes}, {'task': 'binary', 'num_classes': num_classes}, {'task': 'binary', 'num_classes': num_classes}, {'task': 'binary', 'num_classes': num_classes}, {}],
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
print("Model initialized.")
tabular_model.fit(train=train)
print("Model training completed.")
tabular_model.evaluate(test)
print("Model evaluation completed.")

predictions = tabular_model.predict(test=test)
probs = torch.tensor((predictions.iloc[:, 1]).values)
preds = torch.tensor((predictions.iloc[:, 2]).values)
test_target = torch.tensor(test['Label'].values)
confmat = ConfusionMatrix(task='binary', num_classes=num_classes)
res = confmat(preds, test_target)
print(res)
f1 = F1Score(task='binary', num_classes=num_classes)
res = f1(preds, test_target)
print(res)
auroc = AUROC(task="binary")
res = auroc(probs, test_target)
print(res)

savepath = "tabnet-" + time.strftime('%Y%m%d-%H%M')
os.makedirs(savepath)
tabular_model.save_model(savepath)