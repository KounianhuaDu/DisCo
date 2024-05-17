# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import torch
import numpy as np
import datetime
import random
from utils.config import num_feats, num_fields, padding_idxs, item_fields, hist_fields
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score, log_loss



def write_log(w, args):
    file_name = args.log + '/' + args.dataset + '/' + args.model + '/' + datetime.date.today().strftime('%m%d') + \
        f"_LR{args.lr}_WD{args.wd}.log"
    if not os.path.exists(args.log + '/' + args.dataset + '/' + args.model + '/'):
        os.makedirs(args.log + '/' + args.dataset + '/' + args.model + '/')
    t0 = datetime.datetime.now().strftime('%H:%M:%S')
    info = "{} : {}".format(t0, w)
    print(info)
    if not args.test_mode:
        with open(file_name, 'a') as f:
            f.write(info + '\n')


def seed_all(seed, gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)


def evaluate_lp(model, data_loader, device):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for step, (input_ids, label, hist_ids, hist_ratings, hist_mask) in enumerate(data_loader):
            input_ids = input_ids.to(device)
            hist_ids, hist_ratings, hist_mask = hist_ids.to(
                device), hist_ratings.to(device), hist_mask.to(device)
            label = label.to(device)
            logits = model(input_ids, label, hist_ids,
                           hist_ratings, hist_mask)
            logits = logits.squeeze().detach().cpu().numpy().astype('float64')
            label = label.detach().cpu().numpy()
            predictions.append(logits)
            labels.append(label)

    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    acc = ((predictions>0.5)==labels).sum() / predictions.shape[0]
    auc = roc_auc_score(y_score=predictions, y_true=labels)
    logloss = log_loss(y_true=labels, y_pred=predictions,
                       eps=1e-7, normalize=True)
    return auc, acc, logloss


def main(args):
    # set device and seed
    device = torch.device(f"cuda:{args.gpu}" if (
        torch.cuda.is_available() and args.gpu >= 0) else "cpu")
    write_log(f"Device is {device}.", args)
    seed_all(args.seed, device)

    if 'ret' in args.dataloader:
        from dataloaders.ret_loader import load_data
    else:
        from dataloaders.base_loader import load_data
    if 'DeepFM' in args.model and 'xDeepFM' not in args.model:
        from algo.DeepFM import DeepFM
        model = DeepFM(
            num_feat=num_feats[args.dataset],
            num_fields=num_fields[args.dataset],
            padding_idx=padding_idxs[args.dataset],
            embedding_size=args.embedding_size,
            item_fields=item_fields[args.dataset],
            dropout_prob=args.dropout,
            dataset=args.dataset,
            type=args.model.replace('DeepFM', 'p')
        )
        # load data
        train_loader, valid_loader, test_loader = load_data(
            args.dataset, args.batch_size, args.num_workers, args.data_path, semantic_name=args.semantic_name)
    elif 'DCN' in args.model:
        from algo.DCN import DCNV2
        model = DCNV2(
            num_feat=num_feats[args.dataset],
            num_fields=num_fields[args.dataset],
            padding_idx=padding_idxs[args.dataset],
            embedding_size=args.embedding_size,
            item_fields=item_fields[args.dataset],
            dropout_prob=args.dropout,
            dataset=args.dataset,
            type=args.model.replace('DCN', 'p')
        )
        # load data
        train_loader, valid_loader, test_loader = load_data(
            args.dataset, args.batch_size, args.num_workers, args.data_path, semantic_name=args.semantic_name)
    elif 'PNN' in args.model:
        from algo.PNN import IPNN
        model = IPNN(
            num_feat=num_feats[args.dataset],
            num_fields=num_fields[args.dataset],
            padding_idx=padding_idxs[args.dataset],
            embedding_size=args.embedding_size,
            item_fields=item_fields[args.dataset],
            dropout_prob=args.dropout,
            hidden_size=[128, 64],
            dataset=args.dataset,
            type=args.model.replace('PNN', 'p')
        )
        # load data
        train_loader, valid_loader, test_loader = load_data(
            args.dataset, args.batch_size, args.num_workers, args.data_path, semantic_name=args.semantic_name)
    elif 'xDeepFM' in args.model:
        from algo.xDeepFM import xDeepFM
        model = xDeepFM(
            num_feat=num_feats[args.dataset],
            num_fields=num_fields[args.dataset],
            padding_idx=padding_idxs[args.dataset],
            embedding_size=args.embedding_size,
            item_fields=item_fields[args.dataset],
            dropout_prob=args.dropout,
            hidden_size=[128, 64],
            dataset=args.dataset,
            type=args.model.replace('xDeepFM', 'p')
        )
        # load data
        train_loader, valid_loader, test_loader = load_data(
            args.dataset, args.batch_size, args.num_workers, args.data_path, semantic_name=args.semantic_name)
    elif args.model == 'DIN':
        from algo.DIN import DIN
        model = DIN(
            num_feats[args.dataset],
            num_fields[args.dataset],
            padding_idxs[args.dataset],
            item_fields[args.dataset],
            1,
            # hist_fields[args.dataset],
            args.embedding_size,
            args.dropout,
            args.dataset
        )
        # load data
        train_loader, valid_loader, test_loader = load_data(
            args.dataset, args.batch_size, args.num_workers, args.data_path, semantic_name=args.semantic_name)
    elif args.model == 'AutoInt':
        from algo.AutoInt import AutoInt
        model = AutoInt(
            num_feats[args.dataset],
            num_fields[args.dataset],
            padding_idxs[args.dataset],
            item_fields[args.dataset],
            # hist_fields[args.dataset],
            args.embedding_size,
            args.dropout,
            args.dataset
        )
        # load data
        train_loader, valid_loader, test_loader = load_data(
            args.dataset, args.batch_size, args.num_workers, args.data_path, semantic_name=args.semantic_name)
    else:
        raise NotImplementedError
    
    model.apply(weight_init)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    # start training
    write_log('Start training.', args)
    best_auc = 0.0
    kill_cnt = 0
    for epoch in range(args.epochs):
        train_loss = []
        model.train()
        with tqdm(total=len(train_loader), dynamic_ncols=True) as t:
            t.set_description(f'Epoch: {epoch+1}/{args.epochs}')
            for step, (input_ids, label, hist_ids, hist_ratings, hist_mask) in enumerate(train_loader):
                input_ids = input_ids.to(device)
                hist_ids, hist_ratings, hist_mask = hist_ids.to(
                    device), hist_ratings.to(device), hist_mask.to(device)
                label = label.to(device)
                # model forward
                logits = model(input_ids, label, hist_ids,
                               hist_ratings, hist_mask)

                # compute loss
                tr_loss = criterion(logits, label.float())
                train_loss.append(tr_loss.item())

                # backward
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()

                t.update()
                t.set_postfix({
                    'Train loss': f'{tr_loss.item():.4f}'
                })

        train_loss = np.mean(train_loss)

        val_auc, val_acc,  val_logloss = evaluate_lp(
            model, valid_loader, device=device)

        write_log(f"Epoch {epoch}: train loss: {train_loss:.6f}", args)
        write_log(
            f"val AUC: {val_auc:.6f},  val ACC: {val_acc:.6f}, val logloss: {val_logloss:.6f}", args)
        # validate
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(model.state_dict(
            ), f'../saved_models/{args.dataset}/_{args.model}_{args.lr}_{args.wd}.pth')
            kill_cnt = 0
            print("saving model...")
        else:
            kill_cnt += 1
            if kill_cnt >= args.early_stop:
                print(f'Early stop at epoch {epoch}')
                write_log("best epoch: {}".format(best_epoch+1), args)
                break

    # test use the best model
    model.eval()
    model.load_state_dict(torch.load(
        f'../saved_models/{args.dataset}/_{args.model}_{args.lr}_{args.wd}.pth'))
    test_auc, test_acc, test_logloss = evaluate_lp(
        model, test_loader, device=device)
    write_log(f"***********Test Results:************", args)
    write_log(
        f'Test ACC: {test_acc:.4f}',args
    )
    write_log(
        f"Test Auc: {test_auc:.4f}\t Test logloss: {test_logloss:.4f}", args)


def get_args():
    parser = argparse.ArgumentParser()
    # strategy
    parser.add_argument('--strategy', type=str,
                        default='dissem', choices=['dissem'])
    parser.add_argument('--dataloader', type=str, default='ret')
    # dataset
    parser.add_argument('--dataset', type=str,
                        default='ml_1m', choices=['ml_1m', 'AZ_Toys', 'ml_25m'])
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--log', type=str, default='../logs')
    parser.add_argument('--num_workers', type=int, default=8)
    # device
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    # training
    parser.add_argument('--model', type=str, default='DeepFM',)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    # model structure
    parser.add_argument('--embedding_size', type=int, default=32)
    # test
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--semantic_name', type=str, default='13b')
    args = parser.parse_args()
    args.data_path = '../' + args.strategy + '_preprocess'
    return args


if __name__ == '__main__':
    args = get_args()
    write_log(args, args)
    main(args)
