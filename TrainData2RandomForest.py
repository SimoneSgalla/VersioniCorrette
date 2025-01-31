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
    confusion_matrix, classification_report,
)
import numpy as np
import pandas as pd
import time
import os


class TonIotDataset:

    def __init__(self, path, columns=None, types=None):
        self.path = path
        self.dirpath = os.path.dirname(path)

        if columns is None:
            self.__columns = [
                "dow",
                "hour",
                "minute",
                "src_ip",
                "src_port",
                "dst_ip",
                "dst_port",
                "proto",
                "service",
                "duration",
                "src_bytes",
                "dst_bytes",
                "conn_state",
                "missed_bytes",
                "src_pkts",
                "src_ip_bytes",
                "dst_pkts",
                "dst_ip_bytes",
                "dns_query",
                "dns_qclass",
                "dns_qtype",
                "dns_rcode",
                "dns_AA",
                "dns_RD",
                "dns_RA",
                "dns_rejected",
                "ssl_version",
                "ssl_cipher",
                "ssl_resumed",
                "ssl_established",
                "ssl_subject",
                "ssl_issuer",
                "http_trans_depth",
                "http_method",
                "http_uri",
                "http_referrer",
                "http_version",
                "http_request_body_len",
                "http_response_body_len",
                "http_status_code",
                "http_user_agent",
                "http_orig_mime_types",
                "http_resp_mime_types",
                "weird_name",
                "weird_addl",
                "weird_notice",
                "label",
                "type",
            ]
        else:
            self.__columns = columns

        if types is None:
            self.__types = {
                "dow": "string",
                "hour": "int8",
                "minute": "int8",
                "src_ip": "string",
                "src_port": "int32",
                "dst_ip": "string",
                "dst_port": "int32",
                "proto": "string",
                "service": "string",
                "duration": "float64",
                "src_bytes": "int64",
                "dst_bytes": "int64",
                "conn_state": "string",
                "missed_bytes": "int64",
                "src_pkts": "int64",
                "src_ip_bytes": "int64",
                "dst_pkts": "int64",
                "dst_ip_bytes": "int64",
                "dns_query": "string",
                "dns_qclass": "int32",
                "dns_qtype": "int32",
                "dns_rcode": "int32",
                "dns_AA": "string",  # boolean in doc
                "dns_RD": "string",  # boolean in doc
                "dns_RA": "string",  # boolean in doc
                "dns_rejected": "string",  # boolean in doc
                "ssl_version": "string",
                "ssl_cipher": "string",
                "ssl_resumed": "string",  # boolean in doc
                "ssl_established": "string",
                "ssl_subject": "string",
                "ssl_issuer": "string",
                "http_trans_depth": "string",  # on description doc, its type is a number
                "http_method": "string",
                "http_uri": "string",
                "http_referrer": "string",  # not present on description doc
                "http_version": "string",
                "http_request_body_len": "Int64",
                "http_response_body_len": "Int64",
                "http_status_code": "Int16",
                "http_user_agent": "string",  # on description doc this field's type is a number
                "http_orig_mime_types": "string",
                "http_resp_mime_types": "string",
                "weird_name": "string",
                "weird_addl": "string",
                "weird_notice": "string",  # boolean in doc
                "label": "Int8",  # only 0 and 1 as numbers: tag normal and attack records
                "type": "string",
            }
        else:
            self.__types = types

    def load_dataset(self, cols=None):
        if cols is None:
            cols = self.__columns
        self.cols_in_use = cols
        df = pd.read_csv(self.path, sep=",", usecols=cols, dtype=self.__types, nrows=100000)
        return df

    def save_dataset(self, dataframe, filename, dirpath=None):
        if dirpath is None:
            dirpath = self.dirpath
        save_path = dirpath + filename
        dataframe.to_csv(save_path, sep=",", index=False, index_label=False)

# get dataset
filepath = "../Dataset2/Network_dataset_ts_extracted.csv"
ton = TonIotDataset(filepath)
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Loading {filepath}...")
df = ton.load_dataset()
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

###preprocessing

# drop unused label
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Drop 'type' column...")
df.drop(columns=['type'], inplace=True)
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

# separate features from label
start_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Separating label from features...")
y = df['label']
df.drop(columns=['label'], inplace=True)
X = df
end_time = time.time()
time_str = time.strftime("%R")
print(f"<{time_str}> Done. Elapsed {end_time - start_time}s.")

# separate categorical and numerical features
NUMERICAL_FEATURES = X.select_dtypes(include='number').columns.tolist()
CATEGORICAL_FEATURES = X.select_dtypes(exclude='number').columns.tolist()

print(f"Numericals: {len(NUMERICAL_FEATURES)}; Categoricals: {len(CATEGORICAL_FEATURES)}")

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
savepath = "Models/"

for i in scaling_factors:
    print(f"Running model with scale_pos_weight multiplier: {i}")

    # Create the XGBClassifier with scale_pos_weight
    model = RandomForestClassifier()

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

    dump(model, savepath + f"xgboost_model_scale_weight_f{i * 100}.json")

    # Make predictions and compute other scores
    start_prediction_time = time.time()
    y_pred = model.predict(X_test)
    end_prediction_time = time.time()
    num_rows = X_test.shape[0]
    infer_time = (end_prediction_time - start_prediction_time) / num_rows
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print(y_pred_proba.shape)
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
