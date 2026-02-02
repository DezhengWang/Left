import argparse
import torch
import torch.backends
import random
import numpy as np

# from exp.exp_anomaly_detection_contrastive_learning import Exp_Anomaly_Detection
from exp.exp_anomaly_detection import Exp_Anomaly_Detection

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='')

    # basic config
    parser.add_argument('--mode', type=str, default="train", help='status')
    parser.add_argument('--configs_path', type=str, default="./configs/", help='')
    parser.add_argument('--results_save_path', type=str, default='./test_results/', help='')
    parser.add_argument('--model_save_path', type=str, default='./model_results/', help='')

    # data
    parser.add_argument('--root_path', type=str, default='datasets/', help='root path of the data file')
    parser.add_argument('--data', type=str, default='GECCO', help='datasets type')
    parser.add_argument('--data_origin', type=str, default='DADA', help='datasets type')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # anomaly detection
    parser.add_argument('--method', type=str, default='spot', help='')
    parser.add_argument('--t', type=float, nargs='+', default=[0.1], help='')
    parser.add_argument('--metrics', type=str, nargs='+', default=['best_f1', 'auc', 'r_auc', 'vus'], help='')

    # model
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--n_fft', type=int, default=64, help='number of FFT')
    parser.add_argument('--model_class', type=str, default="Left", help='model name')

    args = parser.parse_args()
    args.device = torch.device("cpu")

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # Exp
    exp = Exp_Anomaly_Detection(args, id=0)
    if args.mode == "train":
        exp.train()
        exp.test()
    elif args.mode == "test":
        exp.test()
    elif args.mode == "evaluate":
        exp.evaluate_spot(t=args.t)
    else:
        exp.analysis()

    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()


