import torch
import os.path as osp

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, WikipediaNetwork
from ogb.nodeproppred import PygNodePropPredDataset
import numpy as np

from util import index_to_mask, mask_to_index

def get_dataset(args, split, normalize_features, sparse=True, **kwargs):

    if sparse:
        transform = T.ToSparseTensor()
    else:
        transform=None

    ## fix random seed for data split
    seeds_init = [12232231, 12232432, 2234234, 4665565, 45543345, 454543543, 45345234, 54552234, 234235425, 909099343]
    seeds = []
    for i in range(1, 20):
        seeds = seeds + [a*i for a in seeds_init]
    seed = seeds[split]

    if args.ogb:
        # dataset = get_ogbn_dataset(args.dataset, args.normalize_features, transform=transform)
        # dataset = PygNodePropPredDataset(name=args.dataset,
                                        #  transform=T.ToSparseTensor())
        dataset = PygNodePropPredDataset(name=args.dataset,
                                         transform=transform)
        data = dataset[0]
        # data.adj_t = data.adj_t.to_symmetric()
        split_idx = dataset.get_idx_split()
        # import ipdb; ipdb.set_trace()
        # print(split_idx)
        data.y = data.y.squeeze(1) ## add this for make y [num, 1] to [num]
        if args.const_split:
            train_idx = split_idx['train']
        else:
            if args.fix_num > 0:
                train_idx = random_ogb_splits(data, split_idx['train'], num_classes=dataset.num_classes, seed=seed, num=args.fix_num)
                split_idx['train'] = train_idx
            elif args.proportion > 0:
                train_idx = proportion_ogb_splits(data, split_idx['train'], num_classes=dataset.num_classes, seed=seed, proportion=args.proportion)
                split_idx['train'] = train_idx
        # data.train_mask = index_to_mask(split_idx['train'], data.x.shape[0])
        data.train_mask = index_to_mask(train_idx, data.x.shape[0])
        data.test_mask = index_to_mask(split_idx['test'], data.x.shape[0]) ## add this for convenience
        data.val_mask = index_to_mask(split_idx['valid'], data.x.shape[0]) ## add this for convenience
        return dataset, data, split_idx

    if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
        dataset = get_planetoid_dataset(args.dataset, normalize_features, transform=transform)
        data = dataset[0]
        if args.random_splits > 0:
            if args.active:
                data = activate_planetoid_splits(data, num_classes=dataset.num_classes, seed=seed, num=args.fix_num)
                print('active data split')
            else:
                if args.fix_num > 0:
                    data = random_planetoid_splits(data, num_classes=dataset.num_classes, seed=seed, num=args.fix_num)
                elif args.proportion > 0:
                    data = proportion_planetoid_splits(data, num_classes=dataset.num_classes, seed=seed, proportion=args.proportion)
            print(f'random split {args.dataset} split {split}')

    elif args.dataset == "cs" or args.dataset == "physics":
        dataset = get_coauthor_dataset(args.dataset, normalize_features, transform=transform)
        data = dataset[0]
        data = random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        print(f'random split {args.dataset} split {split}')

    elif args.dataset == "computers" or args.dataset == "photo":
        dataset = get_amazon_dataset(args.dataset, normalize_features, transform=transform)
        data = dataset[0]
        data = random_coauthor_amazon_splits(data, num_classes=dataset.num_classes, seed=seed)
        print(f'random split {args.dataset} split {split}')
    
    elif args.dataset in ["chameleon", "squirrel", "crocodile"]:
        dataset = get_Wiki_dataset(args.dataset, args.normalize_features, transform=transform)
        data = dataset[0]
        # data = random_WebKB_splits(data, num_classes=dataset.num_classes, seed=seed)
        data = load_wiki_fix_split(data, name=args.dataset, seed=split)
        # import ipdb; ipdb.set_trace()

    split_idx = {}
    split_idx['train'] = mask_to_index(data.train_mask)
    split_idx['valid'] = mask_to_index(data.val_mask)
    split_idx['test']  = mask_to_index(data.test_mask)

    return dataset, data, split_idx


def get_transform(normalize_features, transform):
    if transform is not None and normalize_features:
        transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        transform = T.NormalizeFeatures()
    elif transform is not None:
        transform = transform
    return transform

def get_planetoid_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Planetoid(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_coauthor_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Coauthor(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_amazon_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = Amazon(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_ogbn_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = PygNodePropPredDataset(name, path)
    # dataset.transform = get_transform(normalize_features, transform)
    return dataset

def get_Wiki_dataset(name, normalize_features=False, transform=None):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    dataset = WikipediaNetwork(path, name)
    dataset.transform = get_transform(normalize_features, transform)
    return dataset

def random_planetoid_splits(data, num_classes, seed, num):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:num] for i in indices], dim=0)
    print('len(train)', len(train_index))
    rest_index = torch.cat([i[num:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)
    return data

def activate_planetoid_splits(data, num_classes, seed, num):
    # Set new random planetoid splits:
    # * restlabels for training
    # * 500 labels for validation
    # * 500 labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    index = torch.randperm(data.y.shape[0], generator=g)
    val_index = index[:500]
    test_index = index[500:1000]
    train_index = index[1000:]
    # import ipdb; ipdb.set_trace()
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(test_index, size=data.num_nodes)
    return data

def random_ogb_splits(data, idx, num_classes, seed, num):
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y[idx] == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)
    train_index = torch.cat([idx[i[:num]] for i in indices], dim=0)
    # import ipdb; ipdb.set_trace()
    return train_index


def proportion_ogb_splits(data, idx, num_classes, seed, proportion):
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y[idx] == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)
    proportion = proportion * data.y.size(0) / len(idx)
    train_index = torch.cat([idx[i[:int(proportion*len(i))]] for i in indices], dim=0)
    # import ipdb; ipdb.set_trace()
    return train_index


def proportion_planetoid_splits(data, num_classes, seed, proportion):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)
    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:int(proportion*len(i))] for i in indices], dim=0)
    rest_index = torch.cat([i[int(proportion*len(i)):] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    print('len(train)', len(train_index))

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:int(0.5*len(rest_index))], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[int(0.5*len(rest_index)):], size=data.num_nodes)
    return data

def random_coauthor_amazon_splits(data, num_classes, seed):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(seed)

    indices = []
    for i in range(num_classes):
        index = torch.nonzero(data.y == i, as_tuple=False).view(-1)
        index = index[torch.randperm(index.size(0), generator=g)]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)
    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0), generator=g)]
    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data

def load_wiki_fix_split(data, name, seed):
    # seed = 1
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
    masks = np.load(f"{path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{seed}.npz")
    data.train_mask = torch.tensor(masks['train_mask']) > 0
    data.val_mask = torch.tensor(masks['val_mask']) > 0
    data.test_mask = torch.tensor(masks['test_mask']) > 0

    return data