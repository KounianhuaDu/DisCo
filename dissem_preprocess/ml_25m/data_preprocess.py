import pandas as pd
import numpy as np
import json
import os
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

dataset = "ml-25m"
root = f"."
source_dir = os.path.join(root, "raw_data")
target_dir = os.path.join(root, "proc_data")


# Movie data

movie_fields = ["Movie ID", "Movie title", "Movie genre"]
df_movie = pd.read_csv(os.path.join(source_dir, "movies.csv"))
df_movie = df_movie.rename(columns={'movieId': 'Movie ID', 'title': 'Movie title', 'genres': 'Movie genre'})

df_movie['Movie genre'] = df_movie['Movie genre'].apply(lambda x: x.strip().split("|")[0])
print(f"Total number of movies: {len(df_movie)}")
assert len(df_movie["Movie ID"]) == len(set(df_movie["Movie ID"]))

df_movie.head(5)

# Rating data

rating_fields = ["User ID", "Movie ID", "rating", "timestamp", "labels"]
df_ratings = pd.read_csv(os.path.join(source_dir, "ratings.csv"))
df_ratings = df_ratings.rename(columns={'userId': 'User ID', 'movieId': 'Movie ID'})
df_ratings["labels"] = df_ratings["rating"].apply(lambda x: int(x > 3))

print(f"Total number of ratings: {len(df_ratings)}")
df_ratings.head(5)

# Merge df_movie/df_rating into df_data

df_data = pd.merge(df_ratings, df_movie, on=["Movie ID"], how="inner")
df_data.sort_values(by=["timestamp", "User ID", "Movie ID"], inplace=True, kind="stable")
df_data = df_data.reset_index(drop=True)
# df_data = df_data[df_data['Movie title'].apply(lambda x: len(x) <= 70)].reset_index(drop=True)
# print(f"Current movies: {len(set(df_data['Movie title']))}. Deleted movies: {62423-len(set(df_data['Movie title']))}")

# df_data = df_data[df_data['Movie title'].apply(lambda x: character_check(x))].reset_index(drop=True)
# print(f"Current movies: {len(set(df_data['Movie title']))}. Deleted movies: {62423-len(set(df_data['Movie title']))}")

print(f"{len(df_data)}")
df_data.head(5)

# Encode the feature dict for CTR data

def add_to_dict(dict, feature):
    if feature not in dict:
        dict[feature] = len(dict)

field_names = ["User ID", "Movie ID", "Movie title", "Movie genre"]
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

# Re-encode `User ID` and `Movie ID` in `df_data`

re_encode_fields = ["User ID", "Movie ID"]
for field in re_encode_fields:
    df_data[field] = df_data[field].apply(lambda x: feature_dict[field][x])

df_data.head()

# Collect user history

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

print(f"Number of data samples: {len(df_data)}")
print(f"Average history length: {df_data['history ID'].apply(lambda x: len(x)).mean()}")
df_data.head()

# Split & save user history sequence

train_num = int(0.8 * len(df_data))
valid_num = int(0.1 * len(df_data))
test_num = len(df_data) - train_num - valid_num

history_train_id = history_column["ID"][:train_num]
history_valid_id = history_column["ID"][train_num:train_num + valid_num]
history_test_id = history_column["ID"][train_num + valid_num:]

history_train_rating = history_column["rating"][:train_num]
history_valid_rating = history_column["rating"][train_num:train_num + valid_num]
history_test_rating = history_column["rating"][train_num + valid_num:]

history_train_lenth = history_column["hist length"][:train_num]
history_valid_lenth = history_column["hist length"][train_num:train_num + valid_num]
history_test_lenth = history_column["hist length"][train_num + valid_num:]

json.dump(history_train_id, open(os.path.join(target_dir, "history_train_id.json"), "w"), ensure_ascii=False)
json.dump(history_valid_id, open(os.path.join(target_dir, "history_valid_id.json"), "w"), ensure_ascii=False)
json.dump(history_test_id, open(os.path.join(target_dir, "history_test_id.json"), "w"), ensure_ascii=False)


json.dump(history_train_rating, open(os.path.join(target_dir, "history_train_rating.json"), "w"), ensure_ascii=False)
json.dump(history_valid_rating, open(os.path.join(target_dir, "history_valid_rating.json"), "w"), ensure_ascii=False)
json.dump(history_test_rating, open(os.path.join(target_dir, "history_test_rating.json"), "w"), ensure_ascii=False)

json.dump(history_train_lenth, open(os.path.join(target_dir, "history_train_lenth.json"), "w"), ensure_ascii=False)
json.dump(history_valid_lenth, open(os.path.join(target_dir, "history_valid_lenth.json"), "w"), ensure_ascii=False)
json.dump(history_test_lenth, open(os.path.join(target_dir, "history_test_lenth.json"), "w"), ensure_ascii=False)
assert 0
# user_seq = {
#     "history ID": {
#         "train": history_column["ID"][:train_num],
#         "valid": history_column["ID"][train_num:train_num + valid_num],
#         "test": history_column["ID"][train_num + valid_num:],
#     },
#     "history rating": {
#         "train": history_column["rating"][:train_num],
#         "valid": history_column["rating"][train_num:train_num + valid_num],
#         "test": history_column["rating"][train_num + valid_num:],
#     },
#     "history length": {
#         "train": history_column["hist length"][:train_num],
#         "valid": history_column["hist length"][train_num:train_num + valid_num],
#         "test": history_column["hist length"][train_num + valid_num:],
#     },
# }


# json.dump(user_seq, open(os.path.join(target_dir, "user_seq.json"), "w"), ensure_ascii=False)


# Save train/valid/test in parquet format

df_train = df_data[:train_num].reset_index(drop=True)
df_valid = df_data[train_num:train_num + valid_num].reset_index(drop=True)
df_test = df_data[train_num + valid_num:].reset_index(drop=True)

df_train['history ID'] = df_train['history ID'].apply(lambda x: x[-30:])
df_train['history rating'] = df_train['history rating'].apply(lambda x: x[-30:])
df_train.to_csv(os.path.join(target_dir, "train.csv"))

assert len(df_train) == train_num
assert len(df_valid) == valid_num
assert len(df_test) == test_num

print(f"Train num: {len(df_train)}")
print(f"Valid num: {len(df_valid)}")
print(f"Test num: {len(df_test)}")

df_train.to_parquet(os.path.join(target_dir, "train.parquet.gz"), compression="gzip")
df_valid.to_parquet(os.path.join(target_dir, "valid.parquet.gz"), compression="gzip")
df_test.to_parquet(os.path.join(target_dir, "test.parquet.gz"), compression="gzip")


meta_data = {
    "field_names": field_names, # list of field names
    "feature_count": feature_count, # list of field feature counts
    "feature_dict": feature_dict,   # {field:{feat: idx from 0 to field feat counts}}
    "feature_offset": feature_offset, # list [0, ...]
    "movie_id_to_title": movie_id_to_title,
    "num_ratings": len(set(df_data['rating'])),
}

json.dump(meta_data, open(os.path.join(target_dir, "ctr-meta.json"), "w"), ensure_ascii=False, indent=4)


# Convert df_data to CTR data via feature_dict

ctr_X, ctr_Y = [], []
for idx, row in tqdm(df_data.iterrows()):
    ctr_X.append([feature_dict[field][row[field]] if field not in ["Movie ID", "User ID"] else row[field] for field in field_names])
    ctr_Y.append(int(row["labels"]))

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
    for split in user_seq[hist_name]:
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