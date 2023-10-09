import torch
from dataset import get_dataset
from util import index_to_mask, mask_to_index
import cvxpy as cp
import numpy as np
from collections import Counter
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from prop import Propagation
import torch.nn.functional as F
import math


def binary_search(a,eps,xi=1e-6,ub=1):
    mu_l = torch.min(a-1)
    mu_u = torch.max(a)
    while torch.abs(mu_u - mu_l)>xi:
        mu_a = (mu_u + mu_l)/2
        gu = torch.sum(torch.clamp(a-mu_a, 0, ub)) - eps
        gu_l = torch.sum(torch.clamp(a-mu_l, 0, ub)) - eps
        if gu == 0: 
            print('gu == 0 !!!!!')
            break
        if torch.sign(gu) == torch.sign(gu_l):
            mu_l = mu_a
        else:
            mu_u = mu_a
    upper_S_update = torch.clamp(a-mu_a, 0, ub)  
    return upper_S_update

def project(a, epsilon):
    a_proj = torch.clamp(a, 0, 1)
    # import ipdb; ipdb.set_trace()
    if torch.sum(a_proj) > epsilon:
        return binary_search(a, epsilon)
    else:
        return a_proj
    
def project_constraints(t, c, tol=1e-6):
    lambda_low = torch.min(t-1)
    lambda_high = torch.max(t)
    
    while torch.abs(lambda_high - lambda_low) > tol:
        lambda_mid = (lambda_low + lambda_high) / 2
        t_projected = torch.clamp(t - lambda_mid, 0, 1)
        
        if torch.sum(t_projected) > c:
            lambda_low = lambda_mid
        else:
            lambda_high = lambda_mid
            
    return torch.clamp(t - lambda_low, 0, 1)

def grad_t_min(t, S):
    # import ipdb; ipdb.set_trace()
    t1 = t.squeeze()
    product = S * t1
    _, min_indices = torch.max(product, dim=1)
    grad_t = torch.zeros_like(t1)
    grad_t.index_add_(0, min_indices, S[torch.arange(S.shape[0]), min_indices])
    grad_t = grad_t.unsqueeze(1)
    return grad_t

def projected_gradient_descent_all(A, B, Bx, c, epsilon, alpha, max_iter, args, split_idx):
    n = B.shape[0]
    A = A.to(B.device)
    Bx = torch.matmul(A, Bx)
    Bx = torch.matmul(A, Bx)
    t = torch.full((n, 1), epsilon/n).to(B.device)  # initialize t
    t1 = torch.full((n, 1), epsilon/n).to(B.device)
    ones = torch.ones((n,1)).to(B.device)
    max_iter = 100
    dist_matrix = torch.cdist(Bx, Bx, p=2)
    tau = 0.1
    if args.dataset in ['photo', 'computers']:
        tau = 1
    exp_neg_dist = torch.exp(-dist_matrix/tau)
    exp_neg_dist.fill_diagonal_(0)
    S = exp_neg_dist / torch.sum(exp_neg_dist, dim=1, keepdim=True)
    B.fill_diagonal_(0)
    alpha = args.beta
    alpha1 = args.gamma
    alpha2 = args.gamma
    alpha2 = args.beta
    print('c', c)
    for i in range(max_iter):
        grad = grad_t_min(t, S)
        Y = B.matmul(t)
        sc = (Y.t().matmul(ones))[0]/n
        c = sc
        grad1 = 2*B.t().matmul(Y - sc * ones)
        grad2 = grad_t_min(t, B)
        weight = torch.abs(grad1).sum() / torch.abs(grad2).sum()
        weight0 = torch.abs(grad).sum() / torch.abs(grad2).sum()
        t = t + alpha / weight0 * grad - alpha1/weight * grad1 + alpha2 * grad2
        t = project(t, epsilon)
    best_t = None
    best_value = -float('inf')
    ts = (t / t.sum()).squeeze()
    for i in range(5000):
        indices = torch.multinomial(ts, epsilon, replacement=False)
        ts1 = torch.zeros_like(ts)
        ts1[indices] = 1
        product = S * ts1
        value0, _ = torch.max(product, dim=1)
        value0 = torch.sum(value0)
        product = B * ts1
        value2, _ = torch.max(product, dim=1)
        value2 = torch.sum(value2)
        value1 = -torch.var(B.matmul(ts1))
        weight0 = 0 if alpha == 0 else 1
        weight1 = 0 if alpha1 == 0 else 100000
        weight2 = 0 if alpha2 == 0 else 1
        value = weight0 * value0 + weight1 * value1 + weight2 * value2
        if i == 0:
            print('initial', value)
            print('value0', value0, 'value1', value1, 'value2', value2)
            print('value0', weight0 * value0, 'value1', weight1 * value1, 'value2', weight2 * value2)
        if value > best_value:
            best_value = value
            best_t = ts1
    print('best', best_value)

    return best_t
 

class PD_data(object):
    def __init__(self, args):
        super(PD_data, self).__init__()
        self.args = args
        self.prop = Propagation(K=2,
                           alpha=args.pro_alpha,
                           mode='APPNP',
                           cached=True,
                           args=args)

    def get_data(self, split=0):
        args = self.args
        dataset, data, split_idx = get_dataset(args, split, normalize_features=args.normalize_features)
        data.num_classes = data.y.max().item() + 1
        train_mask = data.train_mask.int()
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()
        adj = data.adj_t.to_dense()
        adj = adj + torch.eye(adj.shape[0])
        D = torch.sum(adj, dim=1) ** (-0.5)
        D[D==float('inf')] = 0
        D = torch.diag(D)
        adj = torch.mm(torch.mm(D, adj), D)
        L = torch.eye(adj.shape[0]) - adj
        return data, L, train_mask, split_idx, adj

    def process(self, num=140, split=0, seed=None):
        data, L, train_mask, split_idx, A = self.get_data(split)
        B = torch.eye(data.x.shape[0])
        I = torch.eye(B.shape[0])
        A = A.to(self.args.device)
        I = I.to(self.args.device)
        B = self.args.alpha * torch.inverse(I - (1-self.args.alpha) * A)
        c = self.args.c
        B = B.to(self.args.device)
        X = data.x
        X = X.to(self.args.device)
        t = projected_gradient_descent_all(A, B, X, c, num, 0.001, 100, self.args, split_idx)
        indices = t.long().cpu()
        indices = (indices==1).nonzero().squeeze()
        split_idx['train'] = indices
        print('num_train', indices.shape[0])
        split_idx['valid'] = torch.tensor(range(B.shape[0]))
        split_idx['test'] = torch.tensor(range(B.shape[0]))
        return B, data, split_idx