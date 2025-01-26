from sklearn.ensemble import RandomForestClassifier

from joblib import dump
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix, classification_report
)
import numpy as np
import pandas as pd
import os
import time


def convert_timestamps(df, time_features):
    """Convert Timestamp column into different time features columns listed inside time_features.
    Time features supported: dow, hour, minute, second."""

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d/%m/%Y %H:%M:%S")
    for f in time_features:
        if f == "dow":
            # get day of the week
            df['dow'] = df['Timestamp'].dt.day_name()
        elif f == "hour":
            # get hour of the day
            df['hour'] = df['Timestamp'].dt.hour
        elif f == "minute":
            # get minute of the hour
            df['minute'] = df['Timestamp'].dt.minute
        elif f == "second":
            # get second of the minute
            df['second'] = df['Timestamp'].dt.second
            # drop Timestamp
    df.drop(columns=['Timestamp'], inplace=True)


def binarize_label(df):
    df.rename(columns={"Label": "label_string"}, inplace=True)
    df['Label'] = np.where(df['label_string'] == "Benign", np.zeros(df['label_string'].shape, dtype='int'),
                           np.ones(df['label_string'].shape, dtype='int'))
    df.drop(columns=['label_string'], inplace=True)

# get dataset
dirpath = "drive/MyDrive/Dataset1/"
savepath = "model/"
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
    "Label": "int8",
}
files = os.listdir(dirpath)
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Loading dataset...")
frames = list()
for f in files:
    print(f"Loaded file {f}")
    partial_df = pd.read_csv(dirpath + f, sep=",", usecols=proc_cols)
    frames.append(partial_df)
print("Combining frames...")
df = pd.concat(frames)
del frames
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

###preprocessing

# convert timestamps
times_features = ["dow", "hour", "minute"]
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Replacing 'Timestamp' column with {times_features}...")
convert_timestamps(df, times_features)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

# binarize labels for benign/malicious traffic
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Processing Label column for binary classifier...")
binarize_label(df)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

# replace inf values and deal with NaN values by replacing them with 0
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# separate features from label
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Separating label from features...")
y = df["Label"]
df.drop(columns=["Label"], inplace=True)
X = df
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

# separate categorical and numerical features
NUMERICAL_FEATURES = X.select_dtypes(include="number").columns.tolist()
CATEGORICAL_FEATURES = X.select_dtypes(exclude="number").columns.tolist()

print(
    f"Numericals: {len(NUMERICAL_FEATURES)}; Categoricals: {len(CATEGORICAL_FEATURES)}"
)

# split into train and test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Splitting train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

# target encoding of categorical features
encoder = ce.TargetEncoder(verbose=1, cols=CATEGORICAL_FEATURES, return_df=True)
# encoding train set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Encoding train set...")
encoder.fit(X=X_train, y=y_train)
X_train = encoder.transform(X=X_train, y=y_train)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")
# encoding test set
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Encoding test set...")
encoder.fit(X=X_test, y=y_test)
# documentation suggests not passing y for test data
X_test = encoder.transform(X=X_test)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

###training model
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Training...")
######## test model with different weights ########

# Calculate the scale_pos_weight
num_pos = np.sum(y_train == 1)
num_neg = np.sum(y_train == 0)
scale_pos_weight = num_neg / num_pos
# scaling factors
scaling_factors = [0.25, 0.5, 1, 2, 4]

for i in scaling_factors:
    print(f"Running model with scale_pos_weight multiplier: {i}")

    # Create the XGBClassifier with scale_pos_weight
    model = RandomForestClassifier(
    n_estimators=100,  # Numero di alberi
    max_depth=10,      # Profondità degli alberi
    n_jobs=-1  # Utilizza tutti i core disponibili
)

    # Fit the model
    start_time = time.time()
    time_str = time.strftime("%R")
    print(f"<{time_str}> Training...")
    start_training_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test.values)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    end_training_time = time.time()
    training_time = end_training_time - start_training_time
    end_time = time.time()
    time_str = time.strftime("%R")
    print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

    dump(model, savepath + f"randomForest_model_scale_weight_f{i * 100}.joblib")

    # Make predictions and compute other scores
    start_prediction_time = time.time()
    y_pred = model.predict(X_test)
    end_prediction_time = time.time()
    num_rows = X_test.shape[0]
    infer_time = (end_prediction_time - start_prediction_time) / num_rows
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print results to stdout
    print(f"Scale_pos_weight: {i * scale_pos_weight}")
    print(f"Number of positive samples: {num_pos}")
    print(f"Number of negative samples: {num_neg}")
    print(f"Training time: {training_time}")
    print(f"Average inference time: {infer_time}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"ROC AUC: {roc_auc}")
    print("Confusion Matrix:")
    print(f"True Negatives: {conf_matrix[0][0]}")
    print(f"False Positives: {conf_matrix[0][1]}")
    print(f"False Negatives: {conf_matrix[1][0]}")
    print(f"True Positives: {conf_matrix[1][1]}")

    # Store results
    results = {
        "Scale_pos_weight": i * scale_pos_weight,
        "Training time": training_time,
        "Average inference time": infer_time,
        "Accuracy": accuracy,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "ROC AUC": roc_auc,
        "True Negatives": conf_matrix[0][0],
        "False Positives": conf_matrix[0][1],
        "False Negatives": conf_matrix[1][0],
        "True Positives": conf_matrix[1][1]
    }
    # Save results to a .csv file
    results_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Score'])
    results_df.to_csv(savepath + f"xgboost_resultsscale_weight_f{i * 100}.csv", index=False)