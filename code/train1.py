import torch
import torch.nn.functional as F
import random
import argparse
import time
from torch_geometric.transforms import SIGN
from util import Logger, str2bool, spectral
from dataset import get_dataset
from get_model import get_model
from train_eval import train, test, test_f1
from model import CorrectAndSmooth
from torch_geometric.utils import to_undirected

import numpy as np

import optuna
from myutil import sort_trials
import gc
from active import PD_data

def parse_args():
    parser = argparse.ArgumentParser(description='ALTOPT')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--model', type=str, default='ALTOPT')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--normalize_features', type=str2bool, default=True)
    parser.add_argument('--random_splits', type=int, default=0, help='default use fix split')
    parser.add_argument('--seed', type=int, default=12321312)

    parser.add_argument('--prop', type=str, default='EMP')
    parser.add_argument('--K', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--gamma1', type=float, default=None)
    parser.add_argument('--lambda1', type=float, default=None)
    parser.add_argument('--lambda2', type=float, default=None)
    parser.add_argument('--L21', type=str2bool, default=True)
    parser.add_argument('--alpha', type=float, default=None)

    parser.add_argument('--defense', type=str, default=None)
    parser.add_argument('--ptb_rate', type=float, default=0)
    parser.add_argument('--sort_key', type=str, default='K')
    parser.add_argument('--debug', type=str2bool, default=False)

    parser.add_argument('--loss', type=str, default='CE', help='CE, MSE')
    # parser.add_argument('--loss', type=str, default='MSE', help='CE, MSE')
    parser.add_argument('--LP', type=str2bool, default=False, help='Label propagation')
    # parser.add_argument('--LP', type=str2bool, default=True, help='Label propagation')
    parser.add_argument('--loop', type=int, default=1, help='Iteration number of MLP each epoch')
    parser.add_argument('--fix_num', type=int, default=0, help='number of train sample each class')
    parser.add_argument('--proportion', type=float, default=0, help='proportion of train sample each class')
    parser.add_argument('--has_weight', type=str2bool, default=True)
    parser.add_argument('--noise', type=float, default=0, help='labe noise ratio')
    parser.add_argument('--num_correct_layer', type=int, default=None)
    parser.add_argument('--correct_alpha', type=float, default=None)
    parser.add_argument('--num_smooth_layer', type=int, default=None)
    parser.add_argument('--smooth_alpha', type=float, default=None)
    parser.add_argument('--spectral', type=str2bool, default=False)
    parser.add_argument('--pro_alpha', type=float, default=None)
    parser.add_argument('--const_split', type=str2bool, default=False)
    parser.add_argument('--active', type=str2bool, default=False)
    parser.add_argument('--c', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)

    args = parser.parse_args()
    args.ogb = True if 'ogb' in args.dataset.lower() else False
    return args

def objective(trial):
    args = parse_args()
    args = set_up_trial(trial, args)
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    args.device = device

    if args.random_splits > 0:
        random_split_num = args.random_splits
        print(f'random split {random_split_num} times and each for {args.runs} runs')
    else:
        random_split_num = 1
        print(f'fix split and run {args.runs} times')
    # import ipdb; ipdb.set_trace()
    logger = Logger(args.runs * random_split_num)

    total_start = time.perf_counter()

    ## data split
    macro_f1s = []
    seeds_init = [12232231, 12232432, 2234234, 4665565, 45543345, 454543543, 45345234, 54552234, 234235425, 909099343]
    for split in range(random_split_num):
        if args.model == 'SIGN':
            sparse = False
        else:
            sparse = True
        dataset, data1, split_idx = get_dataset(args, split, defense=False, sparse=sparse, normalize_features=True)
        pdata = PD_data(args)
        B, data, split_idx = pdata.process(num=args.fix_num, seed=seeds_init[split])
        print(split_idx['train'])
        data.psuedo_indices = None
        all_features = data.num_features
        # print('feature', data.num_features)
        args.num_class = data.y.max()+1
        train_idx = split_idx['train']
        print("Data:", data)
        ## add noise
        # mask = data.train_mask
        # num_train = mask.sum()
        # print('num_train', num_train)
        if args.ogb:
            args.num_layers = 3
            args.weight_decay = 0
            args.hidden_channels = 256

        if args.model == 'SIGN':
            data.edge_index = to_undirected(data.edge_index)
            data = SIGN(args.num_layers)(data)
            data.xs = [data.x] + [data[f'x{i}'] for i in range(1, args.num_layers + 1)]
            data.x = None
            for i in range(1, args.num_layers + 1):
                data[f'x{i}'] = None
        else:
            if not isinstance(data.adj_t, torch.Tensor):
                data.adj_t = data.adj_t.to_symmetric()
        data = data.to(device)
        print('load data done')
        model = get_model(args, dataset, all_features).to(device)
        print(model)
        if args.model == 'LP':
            result = test(model, data, split_idx, args=args)
            logger.add_result(split, result)
            continue
        model.reset_parameters()
        print('model reset done')
        t_start = time.perf_counter()
        for run in range(args.runs):
            # print(run)
            runs_overall = split * args.runs + run
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)      
            for epoch in range(1, 1 + args.epochs):
                args.current_epoch = epoch
                loss = train(model, data, train_idx, optimizer, args=args)
                result = test(model, data, split_idx, args=args)
                # logger.add_result(runs_overall, result)
                if args.log_steps > 0:
                    if epoch % args.log_steps == 0:
                        # print(model.FF.min(dim=1))
                        train_acc, valid_acc, test_acc, y_best = result
                        print(f'Split: {split + 1:02d}, '
                              f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_acc:.2f}%, '
                              f'Valid: {100 * valid_acc:.2f}% '
                              f'Test: {100 * test_acc:.2f}%')
            result = [train_acc, valid_acc, test_acc]
            logger.add_result(runs_overall, result)
            macro_f1 = test_f1(model, data, split_idx, args=args)
            macro_f1s.append(macro_f1)
            print('macro_f1', macro_f1)
            t_end = time.perf_counter()
            duration = t_end - t_start
            if args.log_steps > 0:
                print(print(f'Split: {split + 1:02d}, 'f'Run: {run + 1:02d}'), 'time: ', duration)
                logger.print_statistics(runs_overall)
    logger.print_statistics()
    print('mean macro F1', torch.mean(torch.tensor(macro_f1s)))
    train1_acc, valid_acc, train2_acc, test_acc, \
    train1_var, valid_var, train2_var, test_var = logger.best_result(run=None, with_var=True) # to adjust

    trial.set_user_attr("train", train2_var)
    trial.set_user_attr("valid", valid_var)
    trial.set_user_attr("test", test_var)
    return valid_acc

def set_up_trial(trial, args):
    args.lr     = trial.suggest_uniform('lr', 0, 1)
    args.weight_decay     = trial.suggest_uniform('weight_decay', 0, 1)
    args.dropout     = trial.suggest_uniform('dropout', 0, 1)
    args.c     = trial.suggest_uniform('c', 0, 1)
    args.gamma     = trial.suggest_uniform('gamma', 0, 1)
    args.gamma1     = trial.suggest_uniform('gamma1', 0, 1)
    args.beta     = trial.suggest_uniform('beta', 0, 1)
    if args.model == 'GCN':
        pass
    elif args.model == 'GAT':
        pass
    elif args.model == 'SGC':
        pass
    elif args.model == 'SIGN':
        pass
    elif args.model == 'LP':
        args.alpha = trial.suggest_uniform('alpha', 0, 1.00001)
        args.K = trial.suggest_uniform('K', 0, 1000)
    # elif args.model == 'APPNP' or args.prop == 'APPNP':
    elif args.model in ['APPNP', 'IAPPNP', 'MLP', 'APPNPALT']:
        args.alpha     = trial.suggest_uniform('alpha', 0, 1.00001)
        args.pro_alpha = trial.suggest_uniform('pro_alpha', 0, 1.00001)
        args.K = trial.suggest_uniform('K', 0, 1000)

    print('K: ', args.K)
    print('alpha: ', args.alpha)
    print('lr: ', args.lr)
    print('weight_decay: ', args.weight_decay)
    print('dropout: ', args.dropout)
    print('c', args.c)
    print('gamma', args.gamma)
    print('gamma1', args.gamma1)
    print('beta', args.beta)
    return args

def set_up_search_space(args):
    dropout_range = [args.dropout]
    lr_range = [args.lr]
    wd_range = [args.weight_decay]
    alpha_range = [args.alpha]
    lambda1_range = [args.lambda1]
    lambda2_range = [args.lambda2]
    K_range = [args.K]
    loop = [args.loop]
    num_correct_layer_range = [args.num_correct_layer]
    correct_alpha_range = [args.correct_alpha]
    num_smooth_layer_range = [args.num_smooth_layer]
    smooth_alpha_range = [args.smooth_alpha]
    pro_alpha_range = [args.pro_alpha]
    gamma_range = [args.gamma]
    gamma1_range = [args.gamma1]
    beta_range = [args.beta]
    c_range = [args.c]
    if args.loop is None:
        loop = [1]
    if args.dropout is None:
        dropout_range = [0.5, 0.8]

    if args.lr is None:
        # lr_range = [0.01, 0.005, 0.05]  ## 0.005 always worst
        lr_range = [0.01, 0.05] ## 0.05 typically the best but we keep lr fixed as 0.01 since most model use 0.01

    if args.weight_decay is None:
        wd_range = [5e-3, 5e-4, 5e-5]  ## seems 5e-3 is not good in general
    
    if args.c is None:
        c_range = [0.05, 0.08, 0.1, 0.15]

    if args.gamma is None:
        gamma_range = [0.01, 0.05, 0.1, 1]

    if args.gamma1 is None:
        gamma1_range = [0.01, 0.05, 0.1, 1]

    if args.beta is None:
        beta_range = [0.01, 0.05, 0.1]

    if args.model == 'LP':
        if args.alpha is None:
            alpha_range = [0.7, 0.8, 0.9, 1]
        if args.K is None:
            K_range = [1, 3, 5, 10, 30, 50]

    if args.model == 'APPNP' or args.prop == 'APPNP' or args.model == 'IAPPNP' or args.model == 'APPNPALT':
        if args.alpha is None:
            alpha_range = [0, 0.05, 0.1, 0.15]
        if args.pro_alpha is None:
            pro_alpha_range = [0, 0.1, 0.3, 0.5, 0.8, 1]
        if args.K is None:
            K_range = [5, 10]

    search_space = {"lr": lr_range,
                    "weight_decay": wd_range,
                    "lambda1": lambda1_range,
                    "lambda2": lambda2_range,
                    "alpha": alpha_range,
                    "dropout": dropout_range,
                    "K": K_range,
                    'loop': loop,
                    "num_correct_layer": num_correct_layer_range,
                    "correct_alpha": correct_alpha_range,
                    "num_smooth_layer": num_smooth_layer_range,
                    "smooth_alpha": smooth_alpha_range,
                    "pro_alpha": pro_alpha_range,
                    "c": c_range,
                    "gamma": gamma_range,
                    "gamma1": gamma1_range,
                    "beta": beta_range,
                    }
    return search_space

if __name__ == "__main__":
    optuna_total_start = time.perf_counter()

    args = parse_args()
    print('main: ', args)
    search_space = set_up_search_space(args)
    print('search_space: ', search_space)
    num_trial = 1
    for s in search_space.values():
        num_trial = len(s) * num_trial
    print('num_trial: ', num_trial)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(search_space))

    study.optimize(objective, n_trials=num_trial)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    sorted_trial = sort_trials(study.trials, key=args.sort_key)

    for trial in sorted_trial:
        print("trial.params: ", trial.params,
              "  trial.value: ", '{0:.5g}'.format(trial.value),
              "  ", trial.user_attrs)

    test_acc = []
    for trial in sorted_trial:
        test_acc.append(trial.user_attrs['test'])
        # import ipdb; ipdb.set_trace()
    print('test_acc')
    print(test_acc)

    print("Best params:", study.best_params)
    print("Best trial Value: ", study.best_trial.value)
    print("Best trial Acc: ", study.best_trial.user_attrs)

    optuna_total_end = time.perf_counter()
    optuna_total_duration = optuna_total_end - optuna_total_start
    print('optuna total time: ', optuna_total_duration)