import pandas as pd
import numpy as np
import json
import os
import re
import random
import copy
from transformers import set_seed
import hashlib
import json
import pickle as pkl
import h5py
import collections
from tqdm import tqdm


set_seed(42)

dataset = "ml_1m"
root = "./"
source_dir = os.path.join(root, "raw_data")
target_dir = os.path.join(root, "proc_data")


age_dict = {
    1: "under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "above 56"
}

job_dict = {
    0: "other or not specified",
	1: "academic/educator",
	2: "artist",
	3: "clerical/admin",
	4: "college/grad student",
	5: "customer service",
	6: "doctor/health care",
	7: "executive/managerial",
	8: "farmer",
	9: "homemaker",
	10: "K-12 student",
	11: "lawyer",
	12: "programmer",
	13: "retired",
	14: "sales/marketing",
	15: "scientist",
	16: "self-employed",
	17: "technician/engineer",
	18: "tradesman/craftsman",
	19: "unemployed",
	20: "writer",
}


# User data
user_data = []
user_fields = ["User ID", "Gender", "Age", "Job", "Zipcode"]
for line in open(os.path.join(source_dir, "users.dat"), "r").readlines():
    ele = line.strip().split("::")
    user_id, gender, age, job, zipcode = [x.strip() for x in ele]
    assert gender in ["M", "F"], ele
    gender = "male" if gender == "M" else "female"
    age = age_dict[int(age)]
    job = job_dict[int(job)]
    user_data.append([user_id, gender, age, job, zipcode])

df_user = pd.DataFrame(user_data, columns=user_fields)
print(f"Total number of users: {len(df_user)}")
assert len(df_user["User ID"]) == len(set(df_user["User ID"]))

md5_hash = hashlib.md5(json.dumps(df_user.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("df_user", md5_hash)
assert md5_hash == "c08673807000e7373ee40f5229b684db"

# Movie data

movie_data = []
movie_fields = ["Movie ID", "Movie title", "Movie genre"]
for line in open(os.path.join(source_dir, "movies.dat"), "r", encoding="ISO-8859-1").readlines():
    ele = line.strip().split("::")
    movie_id = ele[0].strip()
    movie_title = ele[1].strip()
    movie_genre = ele[2].strip().split("|")[0]
    movie_data.append([movie_id, movie_title, movie_genre])

df_movie = pd.DataFrame(movie_data, columns=movie_fields)
print(f"Total number of movies: {len(df_movie)}")
assert len(df_movie["Movie ID"]) == len(set(df_movie["Movie ID"]))

md5_hash = hashlib.md5(json.dumps(df_movie.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("df_movie", md5_hash)
assert md5_hash == "b8ffd6f7b2bf4cda2986fc272c8b8838"


# Rating data

rating_data = []
rating_fields = ["User ID", "Movie ID", "rating", "timestamp", "labels"]
user_list, movie_list = list(df_user["User ID"]), list(df_movie["Movie ID"])
for line in open(os.path.join(source_dir, "ratings.dat"), "r").readlines():
    ele = [x.strip() for x in line.strip().split("::")] 
    user, movie, rating, timestamp = ele[0], ele[1], int(ele[2]), int(ele[3])
    label = 1 if rating > 3 else 0
    if user in user_list and movie in movie_list:
        rating_data.append([user, movie, rating, timestamp, label])

df_ratings = pd.DataFrame(rating_data, columns=rating_fields)
print(f"Total number of ratings: {len(df_ratings)}")

md5_hash = hashlib.md5(json.dumps(df_ratings.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("df_ratings", md5_hash)
assert md5_hash == "9396b17284bec369afb35652f0cd3093"

# Merge df_user/df_movie/df_rating into df_data

df_data = pd.merge(df_ratings, df_user, on=["User ID"], how="inner")
df_data = pd.merge(df_data, df_movie, on=["Movie ID"], how="inner")

df_data.sort_values(by=["timestamp", "User ID", "Movie ID"], inplace=True, kind="stable")

field_names = ["timestamp", "User ID", "Gender", "Age", "Job", "Zipcode", "Movie ID", "Movie title", "Movie genre", "rating", "labels"]

df_data = df_data[field_names].reset_index(drop=True)

md5_hash = hashlib.md5(json.dumps(df_data.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("df_data", md5_hash)
assert md5_hash == "6c2dba49f202fcd1be74d67fed8182c8"

df_data.head()
# Encode the feature dict for CTR data
def add_to_dict(dict, feature):
    if feature not in dict:
        dict[feature] = len(dict)

field_names = ["User ID", "Gender", "Age", "Job", "Zipcode", "Movie ID", "Movie title", "Movie genre"]
feature_dict = {field : {} for field in field_names}

for idx, row in tqdm(df_data.iterrows()):
    for field in field_names:
        add_to_dict(feature_dict[field], row[field])

feature_count = [len(feature_dict[field]) for field in field_names]

feature_offset = [0]
for c in feature_count[:-1]:
    feature_offset.append(feature_offset[-1] + c)

for field in field_names:
    print(field, len(feature_dict[field]))
    assert len(feature_dict[field]) == len(set(list(df_data[field])))

print("---------------------------------------------------------------")
for f, fc, fo in zip(field_names, feature_count, feature_offset):
    print(f, fc, fo)
print("---------------------------------------------------------------")

md5_hash = hashlib.md5(json.dumps(feature_dict, sort_keys=True).encode("utf-8")).hexdigest()
print("feature_dict", md5_hash)
assert md5_hash == "949a84da2fa28e2362f99ad19af4b889"

md5_hash = hashlib.md5(json.dumps(feature_count, sort_keys=True).encode("utf-8")).hexdigest()
print("feature_count", md5_hash)
assert md5_hash == "357c88e2d97526c96ec8d7fde8b8a301"

md5_hash = hashlib.md5(json.dumps(feature_offset, sort_keys=True).encode("utf-8")).hexdigest()
print("feature_offset", md5_hash)
assert md5_hash == "fce5b7a1c12518a07554f1db56da7dee"

# Re-encode `User ID` and `Movie ID` in `df_data`

re_encode_fields = ["User ID", "Movie ID"]
for field in re_encode_fields:
    df_data[field] = df_data[field].apply(lambda x: feature_dict[field][x])

md5_hash = hashlib.md5(json.dumps(df_data.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("df_data", md5_hash)
assert md5_hash == "52d486db1a5618544829a1ed2a72dc6d"

df_data.head()

# Collect user history (<= 30)

user_history_dict = {
    "ID": {k: [] for k in set(df_data["User ID"])},
    "rating": {k: [] for k in set(df_data["User ID"])},
}
history_column = {
    "ID": [],
    "rating": [],
}
movie_id_to_title = {}

for idx, row in tqdm(df_data.iterrows()):
    user_id, movie_id, rating, title = row["User ID"], row["Movie ID"], row["rating"], row["Movie title"]
    history_column["ID"].append(user_history_dict["ID"][user_id].copy())
    history_column["rating"].append(user_history_dict["rating"][user_id].copy())
    user_history_dict["ID"][user_id].append(movie_id)
    user_history_dict["rating"][user_id].append(rating)
    if movie_id not in movie_id_to_title:
        movie_id_to_title[movie_id] = title

md5_hash = hashlib.md5(json.dumps(history_column, sort_keys=True).encode("utf-8")).hexdigest()
print("history_column", md5_hash)
assert md5_hash == "0f56ebde33e08b6372bb79bd4d67d20c"

md5_hash = hashlib.md5(json.dumps(movie_id_to_title, sort_keys=True).encode("utf-8")).hexdigest()
print("movie_id_to_title", md5_hash)
assert md5_hash == "303c5c11381882c4859765bef07dc730"

# Drop data sample with history length that is less than 5.

df_data["history ID"] = history_column["ID"]
df_data["history rating"] = history_column["rating"]

df_data = df_data[df_data["history ID"].apply(lambda x: len(x)) >= 5].reset_index(drop=True)

history_column["ID"] = [x for x in history_column["ID"] if len(x) >= 5]
history_column["rating"] = [x for x in history_column["rating"] if len(x) >= 5]
history_column["hist length"] = [len(x) for x in history_column["rating"]]

for idx, row in tqdm(df_data.iterrows()):
    assert row["history ID"] == history_column["ID"][idx]
    assert row["history rating"] == history_column["rating"][idx]
    assert len(row["history rating"]) == history_column["hist length"][idx]

del df_data["history ID"]
del df_data["history rating"]

md5_hash = hashlib.md5(json.dumps(df_data.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("df_data", md5_hash)
assert md5_hash == "4129c885ca1fff585b2f8553b8c27ac3"

md5_hash = hashlib.md5(json.dumps(history_column, sort_keys=True).encode("utf-8")).hexdigest()
print("history_column", md5_hash)
assert md5_hash == "ba1ee7ed0ad9dfcce121377a808d2d38"

print(f"Number of data sampels: {len(df_data)}")


# Split & save user history sequence

train_num = int(0.8 * len(df_data))
valid_num = int(0.1 * len(df_data))
test_num = len(df_data) - train_num - valid_num

user_seq = {
    "history ID": {
        "train": history_column["ID"][:train_num],
        "valid": history_column["ID"][train_num:train_num + valid_num],
        "test": history_column["ID"][train_num + valid_num:],
    },
    "history rating": {
        "train": history_column["rating"][:train_num],
        "valid": history_column["rating"][train_num:train_num + valid_num],
        "test": history_column["rating"][train_num + valid_num:],
    },
    "history length": {
        "train": history_column["hist length"][:train_num],
        "valid": history_column["hist length"][train_num:train_num + valid_num],
        "test": history_column["hist length"][train_num + valid_num:],
    },
}

md5_hash = hashlib.md5(json.dumps(user_seq, sort_keys=True).encode("utf-8")).hexdigest()
print("user_seq", md5_hash)
assert md5_hash == "1d763c0028f5294e87183d470028c51e"

json.dump(user_seq, open(os.path.join(target_dir, "user_seq.json"), "w"), ensure_ascii=False)

with open(os.path.join(target_dir, "user_seq.json"), encoding="utf-8") as f:
    md5_hash = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    print("user_seq.json", md5_hash)
    assert md5_hash == "e770d902ea74179852385117363119b5"
    
    
# Save train/valid/test in parquet format

df_train = df_data[:train_num].reset_index(drop=True)
df_valid = df_data[train_num:train_num + valid_num].reset_index(drop=True)
df_test = df_data[train_num + valid_num:].reset_index(drop=True)

md5_hash = hashlib.md5(json.dumps(df_train.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("df_train", md5_hash)
assert md5_hash == "d8864fb649300f030476d143dc8ab9a1"
md5_hash = hashlib.md5(json.dumps(df_valid.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("df_valid", md5_hash)
assert md5_hash == "9b522cb46d5b58b00d59e34efb80d4e1"
md5_hash = hashlib.md5(json.dumps(df_test.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("df_test", md5_hash)
assert md5_hash == "1c5431a227c95d08b72442be77acd8a5"

assert len(df_train) == train_num
assert len(df_valid) == valid_num
assert len(df_test) == test_num

print(f"Train num: {len(df_train)}")
print(f"Valid num: {len(df_valid)}")
print(f"Test num: {len(df_test)}")

df_train.to_parquet(os.path.join(target_dir, "train.parquet.gz"), compression="gzip")
df_valid.to_parquet(os.path.join(target_dir, "valid.parquet.gz"), compression="gzip")
df_test.to_parquet(os.path.join(target_dir, "test.parquet.gz"), compression="gzip")


# Re-read for sanity check

train_dataset = pd.read_parquet(os.path.join(target_dir, "train.parquet.gz"))
valid_dataset = pd.read_parquet(os.path.join(target_dir, "valid.parquet.gz"))
test_dataset = pd.read_parquet(os.path.join(target_dir, "test.parquet.gz"))

md5_hash = hashlib.md5(json.dumps(train_dataset.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("train_dataset", md5_hash)
assert md5_hash == "d8864fb649300f030476d143dc8ab9a1"
md5_hash = hashlib.md5(json.dumps(valid_dataset.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("train_dataset", md5_hash)
assert md5_hash == "9b522cb46d5b58b00d59e34efb80d4e1"
md5_hash = hashlib.md5(json.dumps(test_dataset.values.tolist(), sort_keys=True).encode("utf-8")).hexdigest()
print("test_dataset", md5_hash)
assert md5_hash == "1c5431a227c95d08b72442be77acd8a5"

# Save the meta data for CTR

meta_data = {
    "field_names": field_names,
    "feature_count": feature_count,
    "feature_dict": feature_dict,
    "feature_offset": feature_offset,
    "movie_id_to_title": movie_id_to_title,
    "num_ratings": 5,
}

md5_hash = hashlib.md5(json.dumps(meta_data, sort_keys=True).encode("utf-8")).hexdigest()
print("meta_data", md5_hash)
assert md5_hash == "1d1897258cafa48ae3cfc7dcd0475b8a"

json.dump(meta_data, open(os.path.join(target_dir, "ctr-meta.json"), "w"), ensure_ascii=False)

with open(os.path.join(target_dir, "ctr-meta.json"), encoding="utf-8") as f:
    md5_hash = hashlib.md5(f.read().encode("utf-8")).hexdigest()
    print("ctr-meta.json", md5_hash)
    assert md5_hash == "d588c20f4a921a7e43ba76367c6f32f5"
    
# Convert df_data to CTR data via feature_dict

ctr_X, ctr_Y = [], []
for idx, row in tqdm(df_data.iterrows()):
    ctr_X.append([feature_dict[field][row[field]] if field not in ["Movie ID", "User ID"] else row[field] for field in field_names])
    ctr_Y.append(int(row["labels"]))
    
md5_hash = hashlib.md5(json.dumps(ctr_X, sort_keys=True).encode("utf-8")).hexdigest()
print("ctr_X", md5_hash)
assert md5_hash == "e37d79ccf2dac0f5256359422a3949ef"
md5_hash = hashlib.md5(json.dumps(ctr_Y, sort_keys=True).encode("utf-8")).hexdigest()
print("ctr_Y", md5_hash)
assert md5_hash == "29d1cd69dc8ac31d3dcd2f9d9f92c654"

ctr_X = np.array(ctr_X)
ctr_Y = np.array(ctr_Y)
print("ctr_X", ctr_X.shape)
print("ctr_Y", ctr_Y.shape)
feature_count_np = np.array(feature_count).reshape(1, -1)
assert (ctr_X - feature_count_np <= 0).sum() == ctr_X.shape[0] * ctr_X.shape[1]
assert (ctr_Y == 0).sum() + (ctr_Y == 1).sum() == ctr_Y.shape[0]


# Truncate the user sequence up to 30, i.e., 5 <= length <= 30.

import torch
from torch.nn.utils.rnn import pad_sequence

user_seq_trunc = {
    "history ID": {}, 
    "history rating": {}, 
    "history mask": {}, 
}
for hist_name in user_seq:
    for split in user_seq[hist_name]: # spilt: [train, valid, test]
        if hist_name != "history length":
            user_seq_trunc[hist_name][split] = pad_sequence(
                [torch.tensor(x[-30:]) for x in user_seq[hist_name][split]], 
                batch_first=True, 
            )
        else:
            user_seq_trunc["history mask"][split] = pad_sequence(
                [torch.ones(min(x, 30)) for x in user_seq[hist_name][split]], 
                batch_first=True, 
            )

md5_user_seq_trunc = {}
for hist_name in user_seq_trunc:
    md5_user_seq_trunc[hist_name] = {}
    for split in user_seq_trunc[hist_name]:
        md5_user_seq_trunc[hist_name][split] = user_seq_trunc[hist_name][split].tolist()
        print(hist_name, split, user_seq_trunc[hist_name][split].shape)

md5_hash = hashlib.md5(json.dumps(md5_user_seq_trunc, sort_keys=True).encode("utf-8")).hexdigest()
print("md5_user_seq_trunc", md5_hash)
assert md5_hash == "950c09f5e1a60d58baa70451675e450d"

# Save CTR data & truncated user sequence into one .h5 file

with h5py.File(os.path.join(target_dir, f"ctr.h5"), "w") as hf:
    hf.create_dataset("train data", data=ctr_X[:train_num, :])
    hf.create_dataset("valid data", data=ctr_X[train_num:train_num + valid_num, :])
    hf.create_dataset("test data", data=ctr_X[train_num + valid_num:, :])
    hf.create_dataset("train label", data=ctr_Y[:train_num])
    hf.create_dataset("valid label", data=ctr_Y[train_num:train_num + valid_num])
    hf.create_dataset("test label", data=ctr_Y[train_num + valid_num:])
    for hist_name in user_seq_trunc:
        for split in user_seq_trunc[hist_name]:
            hf.create_dataset(f"{split} {hist_name}", data=user_seq_trunc[hist_name][split])

with h5py.File(os.path.join(target_dir, f"ctr.h5"), "r") as hf:
    assert (ctr_X - np.concatenate([hf["train data"][:], hf["valid data"][:], hf["test data"][:]], axis=0)).sum() == 0
    assert (ctr_Y - np.concatenate([hf["train label"][:], hf["valid label"][:], hf["test label"][:]], axis=0)).sum() == 0
    for hist_name in user_seq_trunc:
        for split in user_seq_trunc[hist_name]:
            assert (user_seq_trunc[hist_name][split] - hf[f"{split} {hist_name}"][:]).sum() == 0
    x = hf["train data"][:]
    assert (x - ctr_X[:train_num, :]).sum() == 0
    print(f"train data: {x.shape}")
    
    x = hf["valid data"][:]
    assert (x - ctr_X[train_num:train_num + valid_num, :]).sum() == 0
    print(f"valid data: {x.shape}")
    
    x = hf["test data"][:]
    assert (x - ctr_X[train_num + valid_num:, :]).sum() == 0
    print(f"test data: {x.shape}")
    
    x = hf["train label"][:]
    assert (x - ctr_Y[:train_num]).sum() == 0
    print(f"train label: {x.shape}")
    
    x = hf["valid label"][:]
    assert (x - ctr_Y[train_num:train_num + valid_num]).sum() == 0
    print(f"valid label: {x.shape}")
    
    x = hf["test label"][:]
    assert (x - ctr_Y[train_num + valid_num:]).sum() == 0
    print(f"test label: {x.shape}")

with open(os.path.join(target_dir, "ctr.h5"), "rb") as f:
    md5_hash = hashlib.md5(f.read()).hexdigest()
    print("ctr.h5", md5_hash)
    assert md5_hash == "26676aa8c9664bfcaf1bca1cc810fc1a"


# Sanity check: ensure each row from DataFrame and ctr is matched

split_names = ["train", "valid", "test"]

dataset = {split: pd.read_parquet(os.path.join(target_dir, f"{split}.parquet.gz")) for split in split_names}

with h5py.File(os.path.join(target_dir, f"ctr.h5"), "r") as hf:
    ctr_data = {
        "data": {split: hf[f"{split} data"][:] for split in split_names},
        "label": {split: hf[f"{split} label"][:] for split in split_names},
    }

for split in split_names:
    for idx, row in tqdm(dataset[split].iterrows()):
        for fi, field in enumerate(field_names):
            if field not in ["Movie ID", "User ID"]:
                assert feature_dict[field][row[field]] == ctr_data["data"][split][idx, fi]
            else:
                assert row[field] == ctr_data["data"][split][idx, fi]
        assert int(row["labels"]) == ctr_data["label"][split][idx]