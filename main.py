import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=str, default='DeepFM')
    parser.add_argument('--path', type=str, default='./logs')
    parser.add_argument('--dataset', type=str, default='ml_1m')
    args = parser.parse_args()
    args.data_dir = 'dissem_preprocess/ml_1m/proc_data'
    
    from utils.draw import get_baseline_max
    print(get_baseline_max(args, args.baseline, args.path))