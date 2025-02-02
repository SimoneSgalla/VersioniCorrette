import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.utils import get_balanced_sampler, get_class_weighted_cross_entropy

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
# convert integer types to float, resolve future warning of implicit casting of TabTransformer
for col in NUMERICAL_FEATURES:
    is_int = pd.api.types.is_integer_dtype(df[col])
    if is_int:
        df[col] = df[col].astype('float64')
NUMERICAL_FEATURES.remove('Label')

# split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
train, test = train_test_split(df, stratify=df["Label"], test_size=0.2, random_state=42)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time-start_time}s.")
num_classes = len(set(train["Label"].values.ravel()))
num_pos = np.sum(train["Label"] == 1)
num_neg = np.sum(train["Label"] == 0)
print("Train stats:")
print(f"Number of classes: {num_classes}")
print(f"Number of positive samples: {num_pos}")
print(f"Number of negative samples: {num_neg}")

# weight batches for imbalanced dataset
# class_sample_count = np.array(
#     [len(np.where(train["Label"] == t)[0]) for t in np.unique(train["Label"])])
# weight = 1. / class_sample_count
# samples_weight = np.array([weight[t] for t in (train["Label"].astype(int)).to_numpy()])

# samples_weight = torch.from_numpy(samples_weight)
# samples_weight = samples_weight.double()
# print(samples_weight)
# print(len(samples_weight))
# sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=False)
sampler = get_balanced_sampler(train['Label'].values.ravel())
weighted_loss = get_class_weighted_cross_entropy(train["Label"].values.ravel(), mu=1.0)

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
    batch_size=1048576*8,
    max_epochs=1,
    early_stopping="valid_loss",  # Monitor valid_loss for early stopping
    early_stopping_mode="min",  # Set the mode as min because for val_loss, lower is better
    early_stopping_patience=5,  # No. of epochs of degradation training will wait before terminating
    checkpoints="valid_loss",  # Save best checkpoint monitoring val_loss
    load_best=True,  # After training, load the best checkpoint
    profiler='simple', # https://pytorch-lightning.readthedocs.io/en/1.5.10/advanced/profiler.html
)

optimizer_config = OptimizerConfig()

head_config = LinearHeadConfig(
    layers="",  # No additional layer in head, just a mapping layer to output_dim
    dropout=0.1,
    initialization="kaiming",
).__dict__ # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)

model_config = TabNetModelConfig(
     task="classification",
    learning_rate=1e-3,
    head="LinearHead",  # Linear Head
    head_config=head_config,  # Linear Head Config
    metrics=['accuracy', 'f1_score', 'precision', 'recall', 'auroc'], #found in Lib/site-packages/torchmetrics/functional/init.py
    metrics_prob_input=[False, False, False, False, True],
    metrics_params=[{'task': 'binary', 'num_classes': num_classes}, {'task': 'binary', 'num_classes': num_classes}, {'task': 'binary', 'num_classes': num_classes}, {'task': 'binary', 'num_classes': num_classes}, {}],
)


tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
# training_type = ["simple", "sampler", "weighted"]
training_type = "weighted"

time_str = time.strftime("%R")
print(f"<{time_str}> Model initialized.")
start_training_time = time.time()
if training_type == "simple":
    tabular_model.fit(train=train, validation=test)
elif training_type == "sampler":
    tabular_model.fit(train=train, validation=test, sampler=sampler)
elif training_type == "weighted":
    tabular_model.fit(train=train, validation=test, loss=weighted_loss)
end_training_time = time.time()
training_time = end_training_time - start_training_time
time_str = time.strftime("%R")
print(f"<{time_str}> Model training completed. Elapsed {training_time}s.")
# training can generate user warnings along the lines of "no positive/negative in target"
# especially if sampler or weight aren't used
# that is caused by some metrics being calculated at the end of each batch

time_str = time.strftime("%R")
print(f"<{time_str}>Saving model...")
savepath = "Models/ids2018/tabtransformer-" + time.strftime('%Y%m%d-%H%M') + "/"
os.makedirs(savepath)
tabular_model.save_model(savepath)
time_str = time.strftime("%R")
print(f"<{time_str}>Model saved.")

time_str = time.strftime("%R")
print(f"<{time_str}>Evaluating model...")
start_evaluation_time = time.time()
scores = tabular_model.evaluate(test)
end_evaluation_time = time.time()
evaluation_time = end_evaluation_time - start_evaluation_time
time_str = time.strftime("%R")
print(f"<{time_str}>Model evaluation completed. Elapsed {evaluation_time}s.")

time_str = time.strftime("%R")
print(f"<{time_str}>Test predictions (obtain average inference time)...")
test_rows = test.shape[0]
start_predictions_time = time.time()
# predictions columns: [benign_probability, attack_probability, predicted_label]
predictions = tabular_model.predict(test=test)
end_predictions_time = time.time()
predictions_time = end_predictions_time - start_predictions_time
time_str = time.strftime("%R")
print(f"<{time_str}>Predictions completed. Elapsed {predictions_time}s.")
average_inference_time = predictions_time / test_rows
print(f"Average inference time: {average_inference_time}s.")

print("---Other metrics---")
probs = torch.tensor((predictions.iloc[:, 1]).values)
preds = torch.tensor((predictions.iloc[:, 2]).values)
test_target = torch.tensor(test['Label'].values)
confmat = ConfusionMatrix(task='binary', num_classes=num_classes)
confmat_res = confmat(preds, test_target)
print("Confusion Matrix:")
print(f"True Negatives: {confmat_res[0, 0].item()}")
print(f"False Positives: {confmat_res[0, 1].item()}")
print(f"False Negatives: {confmat_res[1, 0].item()}")
print(f"True Positives: {confmat_res[1, 1].item()}")
(scores[0]).update({
    "TN": confmat_res[0, 0].item(),
    "FP": confmat_res[0, 1].item(),
    "FN": confmat_res[1, 0].item(),
    "TP": confmat_res[1, 1].item(),
    "avg_inf_time": average_inference_time,
    "training_time": training_time,
})

f1 = F1Score(task='binary', num_classes=num_classes)
f1_res = f1(preds, test_target)
print(f"F1_Score: {f1_res.item()}")
auroc = AUROC(task="binary")
auroc_res = auroc(probs, test_target)
print(f"AUROC_Score: {auroc_res.item()}")

# Save scores to a .txt file
scores_txt_path = savepath + "scores.txt"
with open(scores_txt_path, "w") as f:
    for metric, score in (scores[0]).items():
        f.write(f"{metric}: {score}\n")

# Save scores to a .csv file
scores_csv_path = savepath + "scores.csv"
scores_df = pd.DataFrame(list((scores[0]).items()), columns=["Metric", "Value"])
scores_df.to_csv(scores_csv_path, index=False)