import argparse
import re
import os
from os import walk
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DisDeepFM')
    parser.add_argument('--baseline', type=str, default='DeepFM')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dataset', type=str, default='ml_1m')
    return parser.parse_args()


def is_digit(num):
    if num.count('.') == 1:
        left, right = num.split('.')[0], num.split('.')[1]
        return left.isdigit() and right.isdigit()
    return False


def get_baseline_max(args, baseline, path):
    best_auc = 0.5
    best_logloss = 99999
    best_param = None
    path = os.path.join(path, args.dataset, baseline)
    for cur_path, sub_folder, files in walk(path):
        for file in files:
            first_line = open(os.path.join(cur_path, file), 'r').readlines()[0]
            last_line = open(os.path.join(cur_path, file), 'r').readlines()[-1]
            auc = last_line[21:27]
            log_loss = last_line[43:49]
            if not is_digit(auc) or not is_digit(log_loss):
                continue
            auc = float(last_line[21:27])
            log_loss = float(last_line[43:49])
            if auc > best_auc or (auc == best_auc and log_loss < best_logloss):
                best_auc = auc
                best_logloss = log_loss
                best_param = first_line
    return best_auc, best_logloss, best_param


def get_max(args, alpha, learning_rate):
    best_auc = 0.5
    best_logloss = 99999
    path = os.path.join('../logs', args.dataset, args.model, str(alpha))
    total = 0
    for cur_path, sub_folder, files in walk(path):
        for file in files:
            lr = float(file.split('-')[-1][2:-4])
            if lr == learning_rate:
                last_line = open(os.path.join(cur_path, file),
                                 'r').readlines()[-1]
                auc = last_line[21:27]
                log_loss = last_line[43:49]
                if not is_digit(auc) or not is_digit(log_loss):
                    continue
                total += 1
                auc = float(last_line[21:27])
                log_loss = float(last_line[43:49])
                if auc > best_auc or (auc == best_auc and log_loss < best_logloss):
                    best_auc = auc
                    best_logloss = log_loss
    print(
        f'Total results number of {alpha},{learning_rate}: {total} with best auc {best_auc} and logloss {best_logloss}')
    return best_auc, best_logloss


def draw_disentagle(args):
    all_lr = [[], [], [], []]  # 1e-4,3e-4,5e-4,1e-3
    lrs = [1e-4, 3e-4, 5e-4, 1e-3]
    alphas = [0.02, 0.05, 0.2, 0.5]
    for i, lr in enumerate(lrs):
        for alpha in alphas:
            auc, _ = get_max(args, alpha, lr)
            all_lr[i].append(auc)
    for i, cur_lr in enumerate(all_lr):
        plt.plot(alphas, cur_lr, 'o-', label=str(lrs[i]))
    baseline_auc, _ = get_baseline_max(args.baseline)
    plt.plot(alphas, [baseline_auc]*len(alphas), '-', label=args.baseline)

    plt.title(args.model)
    plt.xlabel('Alpha')
    plt.ylabel('Auc')

    plt.legend()
    plt.savefig(f'../logs/{args.dataset}/{args.model}/results/{args.model}_VS_{args.baseline}.png')


def main(args):
    draw_disentagle(args)


if __name__ == '__main__':
    args = get_args()
    main(args)
