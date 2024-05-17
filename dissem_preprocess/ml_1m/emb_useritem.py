import pandas as pd
import re
import os
import copy
import torch
import numpy as np
import pickle as pkl
import argparse
import json
from fastchat.model import load_model, get_conversation_template, add_model_args
from sentence_transformers import SentenceTransformer
from config import *
from tqdm import tqdm

cur_embed = None
embeds = []


def hook(module, input, output):
    global cur_embed, embeds
    input = input[0].cpu().detach().numpy()
    cur_embed = input

def get_bert_embed(model, msg_iter):
    return model.encode(msg_iter)
    

@torch.inference_mode()
def get_embed(model, tokenizer, msg_iter):
    global cur_embed, embeds

    # Start inference.
    for txt in tqdm(msg_iter):
        conv = get_conversation_template(args.model_path)
        conv.append_message(conv.roles[0], txt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        cur_embed = None
        input_ids = tokenizer([prompt]).input_ids

        output_ids = model.generate(
            torch.as_tensor(input_ids).to(args.device),
            do_sample=True,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )
        if args.pooling == "last":
            cur_embed = cur_embed[0, len(input_ids[0])-1]
        elif args.pooling == "average":
            cur_embed = cur_embed[0, :len(input_ids[0])].mean(axis=0)
        print(txt, cur_embed)
        haha=0
        input(haha)

        embeds.append(cur_embed)

    returned_embed = copy.deepcopy(np.stack(embeds))
    cur_embed = None
    embeds.clear()
    return returned_embed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--pooling", type=str,
                        default="average", help="average/last")
    parser.add_argument("--embed_dir", type=str, default="./PLM_data")
    parser.add_argument('--debug', action='store_true')
    # parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument('--model_name',type=str,default='13b')
    args = parser.parse_args()
    args.model_path = 'lmsys/vicuna-7b-v1.3' if args.model_name == '7b' else ('lmsys/vicuna-13b-v1.3' if args.model_name == '13b' else 'all-MiniLM-L12-v2')

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2
    assert args.pooling in ['average', 'last'], 'Pooling type error'

    data_dir = './proc_data'
    user_path = './raw_data/users.dat'
    item_path = './raw_data/movies.dat'
    rating_path = './raw_data/ratings.dat'

    # get item encode
    item_dict = {}
    itemIdxtoName = {}
    for line in open(item_path, 'r',encoding='ISO-8859-1').readlines():
        ele = line.strip().split('::')
        item_id, name, genre = [x.strip() for x in ele]
        item_id = int(item_id)
        matchObj = re.match(r'(.*) .(....).', name, re.M | re.I)
        title, year = matchObj.group(1), matchObj.group(2)
        itemIdxtoName[item_id] = title
        item_dict[item_id] = f'Here is a movie, name is {name}, genre is {genre}'
    print(item_dict[250])
    # save item prompt dict
    with open(os.path.join(args.embed_dir, 'item_prompt.pkl'), 'wb') as f:
        pkl.dump(item_dict, f)
    print('Item prompt saved')

    # get user encode
    user_dict = {}
    for line in open(user_path, 'r').readlines():
        ele = line.strip().split('::')
        user_id, gender, age, job, zipcode = [x.strip() for x in ele]
        user_id = int(user_id)
        user_dict[user_id] = f'This is a user, gender is {user_feature[gender]}, age is {user_feature["age"][age]}, job is {user_feature["occupation"][job]}.'
    print(user_dict[250])

    # save user prompt dict
    with open(os.path.join(args.embed_dir, 'user_prompt.pkl'), 'wb')as f:
        pkl.dump(user_dict, f)
    print('User prompt saved')

    # Build the mapping from encoded ID to datasetID
    meta_data = json.load(
        open(os.path.join(data_dir, 'ctr-meta.json'), 'r')
    )

    user_id_dict = meta_data['feature_dict']['User ID']
    item_id_dict = meta_data['feature_dict']['Movie ID']
    # args.device = f"cuda:{args.gpu_id}" if (
    #     torch.cuda.is_available() and args.gpu_id >= 0) else "cpu"
    print(f'Device is {args.device}')

    # Load model.
    if args.model_name == 'BERT':
        model = SentenceTransformer('all-MiniLM-L12-v2')
    else:
        model, tokenizer = load_model(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.cpu_offloading,
            # revision=args.revision,
            debug=args.debug,
        )
        # Get embedding in lm_head_layer
        model.lm_head.register_forward_hook(hook)
        model.to(torch.device(args.device))

    print('*************Model loaded***************')
    print(f'Model.device is {model.device}')
    # store embeddings
    os.makedirs(args.embed_dir, exist_ok=True)
    if args.model_name == 'BERT':
        user_embedding = get_bert_embed(model,list(user_dict.values()))
        item_embedding = get_bert_embed(model,list(item_dict.values()))
        
    else:
        user_embedding = get_embed(model, tokenizer, list(user_dict.values()))
        item_embedding = get_embed(model, tokenizer, list(item_dict.values()))

    # user embedding search table
    user_index = np.zeros([6041, 4096]) if args.model_name == '7b' else (np.zeros([6041, 5120]) if args.model_name == '13b' else np.zeros([6041,384]))
    # item_embedding search table
    item_index = np.zeros([3706, 4096]) if args.model_name == '7b' else (np.zeros([3706, 5120]) if args.model_name == '13b' else np.zeros([3706,384]))
    for index, user_emb in zip(user_dict.keys(), user_embedding):
        if str(index) in user_id_dict.keys():
            user_index[user_id_dict[str(index)]] = user_emb
    for index, item_emb in zip(item_dict.keys(), item_embedding):
        if str(index) in item_id_dict.keys():
            item_index[item_id_dict[str(index)]] = item_emb
    print(item_embedding.shape, user_embedding.shape)
    # save user and item embedding tables
    user_saved_name = f'user_emb_{args.model_name}'
    item_saved_name = f'item_emb_{args.model_name}'
    np.save(os.path.join(args.embed_dir, user_saved_name), user_index)
    np.save(os.path.join(args.embed_dir, item_saved_name), item_index)
    print('Embedding saved')
    # pca
    print('PCAing!!')
    from sklearn.decomposition import PCA
    from sklearn import preprocessing
    pca = PCA(n_components=512)
    pca_emb = pca.fit_transform(item_index)
    pca_emb = preprocessing.normalize(pca_emb,'l2')
    print(f'After PCA dimension reduce from {item_index.shape} to {pca_emb.shape}')
    np.save(os.path.join(args.embed_dir, f'pca_{args.model_name}'),pca_emb)
    
