import gzip
from pprint import pprint
from tqdm import tqdm
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
from urllib.request import urlopen
import math
import html
from collections import Counter
from pprint import pprint

set_seed(42)

dataset_name = f"AZ-Toys"
root = f"./"
source_dir = os.path.join(root, "raw_data")
target_dir = os.path.join(root, "proc_data")

# Load the meta data for each product

df_items = pd.read_json(
    os.path.join(source_dir, "meta_Toys_and_Games.json.gz"), 
    lines=True, 
    compression="gzip", 
    # dtype=False, 
)

df_items["Item ID"] = df_items["asin"]
df_items["Category"] = df_items["category"]
df_items["Title"] = df_items["title"]
df_items["Brand"] = df_items["brand"]
item_fields = ["Item ID", "Category", "Title", "Brand"]
df_items = df_items[item_fields]
print(df_items.head())



# Convert `Category`

def convert_category(x):
    if isinstance(x, list):
        x = [c for c in x if c != "Toys & Games"]
        x = f"Toys & Games, {x[0].strip()}" if len(x) else "Toys & Games"
    else:
        x = "Toys & Games"
    return x

df_items["Category"] = df_items["Category"].apply(convert_category)


print(set(df_items["Category"]))


# Convert `Title`

from string import ascii_letters, digits, punctuation, whitespace

def filter_title(x):
    x = html.unescape(x)
    x = x.replace("“", "\"")
    x = x.replace("”", "\"")
    x = x.replace("‘", "'")
    x = x.replace("’", "'")
    x = x.replace("–", "-")
    x = x.replace("\n", " ")
    x = x.replace("\r", " ")
    x = x.replace("…", "")
    x = x.replace("‚", ",")
    x = x.replace("´", "'")
    x = x.replace("&ndash;", "-")
    x = x.replace("&lt;", "<")
    x = x.replace("&gt;", ">")
    x = x.replace("&amp;", "&")
    x = x.replace("&quot;", "\"")
    x = x.replace("&nbsp;", " ")
    x = x.replace("&copy;", "©")
    x = x.replace("″", "\"")
    x = x.replace("【", "[")
    x = x.replace("】", "]")
    x = x.replace("—", "-")
    x = x.replace("−", "-")
    for ci in [127, 174, 160, 170, 8482, 188, 162, 189, 8594, 169, 235, 168, 957, 12288, 8222, 179, 190, 173, 186, 8225]:
        x = x.replace(chr(ci), " ")
    while "  " in x:
        x = x.replace("  ", " ")
    x = x.strip()
    for letter in x:
        if letter not in ascii_letters + digits + punctuation + whitespace:
            return ascii_letters + digits + punctuation + whitespace # Delete Flag
    if len(set(x) & set(ascii_letters)) == 0 or len(x) < 3:
        return ascii_letters + digits + punctuation + whitespace # Delete Flag
    if len(x) > 150:
        x = x[:150]
        x = " ".join(x.strip().split(" ")[:-1])
        while x[-1] not in ascii_letters + digits:
            x = x[:-1]
    return x

df_items["Title"] = df_items["Title"].apply(filter_title)
df_items = df_items[df_items["Title"] != ascii_letters + digits + punctuation + whitespace]


print(len(set(df_items["Title"])))

# Convert `Brand`

from string import ascii_letters, digits, punctuation, whitespace

xx, yy = [], []

def convert_brand(x):
    x = html.unescape(x)
    x = x.replace("“", "\"")
    x = x.replace("”", "\"")
    x = x.replace("‘", "'")
    x = x.replace("’", "'")
    x = x.replace("–", "-")
    x = x.replace("\n", " ")
    x = x.replace("\r", " ")
    x = x.replace("…", "")
    x = x.replace("‚", ",")
    x = x.replace("´", "'")
    x = x.replace("&ndash;", "-")
    x = x.replace("&lt;", "<")
    x = x.replace("&gt;", ">")
    x = x.replace("&amp;", "&")
    x = x.replace("&quot;", "\"")
    x = x.replace("&nbsp;", " ")
    x = x.replace("&copy;", "©")
    x = x.replace("″", "\"")
    x = x.replace("【", "[")
    x = x.replace("】", "]")
    x = x.replace("—", "-")
    x = x.replace("−", "-")
    for ci in [127, 174, 160, 170, 8482, 188, 162, 189, 8594, 169, 235, 168, 957, 12288, 8222, 179, 190, 173, 186, 8225]:
        x = x.replace(chr(ci), " ")
    while "  " in x:
        x = x.replace("  ", " ")
    x = x.strip()
    for letter in x:
        if letter not in ascii_letters + digits + punctuation + whitespace:
            return ascii_letters + digits + punctuation + whitespace # Delete Flag
    if len(set(x) & set(ascii_letters)) == 0 or len(x) < 3:
        return ascii_letters + digits + punctuation + whitespace # Delete Flag
    if len(x) > 150:
        x = x[:150]
        x = " ".join(x.strip().split(" ")[:-1])
        while x[-1] not in ascii_letters + digits:
            x = x[:-1]
    return x

df_items["Brand"] = df_items["Brand"].apply(convert_brand)
df_items = df_items[df_items["Brand"] != ascii_letters + digits + punctuation + whitespace]


print(len(set(df_items["Brand"])))

# Load ratings

df_ratings = pd.read_json(
    os.path.join(source_dir, "Toys_and_Games_5.json.gz"), 
    lines=True, 
    compression="gzip", 
    # dtype=False, 
)

df_ratings["User ID"] = df_ratings["reviewerID"]
df_ratings["Item ID"] = df_ratings["asin"]
df_ratings["timestamp"] = df_ratings["unixReviewTime"]
df_ratings["rating"] = df_ratings["overall"]

df_ratings = df_ratings[["User ID", "Item ID", "rating", "timestamp"]]

print(len(df_ratings))


# Merge df_ratings & df_items into df_data

df_data = pd.merge(df_ratings, df_items, on=["Item ID"], how="inner")

df_data["labels"] = df_data["rating"].apply(lambda x: 1 if x > 3 else 0)

df_data.sort_values(by=['timestamp', 'User ID', 'Item ID'], inplace=True, kind='stable')

field_names = ["User ID", "Item ID", "Category", "Title", "Brand", "rating", "labels"]

df_data = df_data[field_names].reset_index(drop=True)

df_data = df_data.drop_duplicates().reset_index(drop=True)

print(len(df_data))

print(df_data["labels"].mean())

# Meta data for CTR

field_names = ["User ID", "Item ID", "Category", "Title", "Brand",]

def add_to_dict(dict, feature):
    if feature not in dict:
        dict[feature] = len(dict)

feature_dict = {field : {} for field in field_names}

for field in tqdm(field_names):
    for f in sorted(set(df_data[field])):
        add_to_dict(feature_dict[field], f)
        
feature_count = [len(feature_dict[field]) for field in field_names]

for field in field_names:
    assert len(feature_dict[field]) == len(set(df_data[field]))

feature_offset = [0]
for c in feature_count[:-1]:
    feature_offset.append(feature_offset[-1] + c)

for f, fc, fo in zip(field_names, feature_count, feature_offset):
    print(f, fc, fo)



# Re-encode user/item ID
print(df_data.columns)
for field in ["User ID", "Item ID"]:
    df_data[field] = df_data[field].apply(lambda x: feature_dict[field][x])
print(df_data.head())


# Collect user history (<=30)

user_history_dict = {
    'ID': {k:[] for k in set(df_data['User ID'])},
    'rating': {k: [] for k in set(df_data['User ID'])}
}

history_column = {
    "ID": [],
    "rating": [],
}

item_id_to_title = {}

for idx, row in tqdm(df_data.iterrows()):
    user_id, item_id, rating, title = row["User ID"], row["Item ID"], row["rating"], row["Title"]
    history_column["ID"].append(user_history_dict["ID"][user_id].copy())
    history_column["rating"].append(user_history_dict["rating"][user_id].copy())
    user_history_dict["ID"][user_id].append(item_id)
    user_history_dict["rating"][user_id].append(rating)
    if item_id not in item_id_to_title:
        item_id_to_title[item_id] = title


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

print(f"Number of data samples: {len(df_data)}")

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

json.dump(user_seq, open(os.path.join(target_dir, "user_seq.json"), "w"), ensure_ascii=False)

df_train = df_data[:train_num].reset_index(drop=True)
df_valid = df_data[train_num:train_num + valid_num].reset_index(drop=True)
df_test = df_data[train_num + valid_num:].reset_index(drop=True)

assert len(df_train) == train_num
assert len(df_valid) == valid_num
assert len(df_test) == test_num

print(f"Train num: {len(df_train)}")
print(f"Valid num: {len(df_valid)}")
print(f"Test num: {len(df_test)}")

df_train.to_parquet(os.path.join(target_dir, "train.parquet.gz"), compression="gzip")
df_valid.to_parquet(os.path.join(target_dir, "valid.parquet.gz"), compression="gzip")
df_test.to_parquet(os.path.join(target_dir, "test.parquet.gz"), compression="gzip")

    
# Save the meta data for CTR

meta_data = {
    'field_names': field_names,
    'feature_count': feature_count,
    'feature_dict': feature_dict,
    'feature_offset': feature_offset,
    'item_id_to_title':item_id_to_title,
    'num_ratigs':5
}

json.dump(meta_data, open(os.path.join(target_dir, "ctr-meta.json"), "w"), ensure_ascii=False)


ctr_X, ctr_Y = [], []
for idx, row in tqdm(df_data.iterrows()):
    ctr_X.append([feature_dict[field][row[field]] if field not in ["Item ID", "User ID"] else row[field] for field in field_names])
    ctr_Y.append(int(row["labels"]))
    

ctr_X = np.array(ctr_X)
ctr_Y = np.array(ctr_Y)
print("ctr_X", ctr_X.shape)
print("ctr_Y", ctr_Y.shape)
feature_count_np = np.array(feature_count).reshape(1, -1)
assert (ctr_X - feature_count_np <= 0).sum() == ctr_X.shape[0] * ctr_X.shape[1]
assert (ctr_Y == 0).sum() + (ctr_Y == 1).sum() == ctr_Y.shape[0]


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
    
