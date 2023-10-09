import torch
import torch_geometric
import torch.nn.functional as F
import sklearn
from ogb.nodeproppred import Evaluator
import  math
import torch.nn as nn
from sklearn.metrics import f1_score

def cross_entropy(pred, target):
    pred = torch.log(pred)
    return torch.mean(torch.sum(-target * pred, 1))

def cross_entropy1(pred, target):
    # pred = pred.clamp(min=1e-8)
    pred = torch.log(pred)
    return -torch.sum(target * pred, 1)

def KL(pred, target):
    return F.kl_div(pred.log(), target)


def train_altopt_PTA(model, data, train_idx, optimizer, args=None):
    model.train()
    label = model.FF
    train_mask = data.train_mask
    optimizer.zero_grad()
    y_hat = model(data=data)
    # import ipdb
    # ipdb.set_trace()
    gamma = math.log(1 + (args.current_epoch-1)/100)
    y_hat_con = torch.detach(torch.softmax(y_hat, dim=-1))
    # loss = - torch.sum(torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(label, y_hat_con ** gamma))) / args.num_class  # PTA
    # loss = - torch.sum(
        # torch.mul(torch.log_softmax(y_hat, dim=-1), torch.mul(label, y_hat_con))) / args.num_class
    loss = - torch.sum(
        torch.mul(torch.log_softmax(y_hat, dim=-1), label)) / args.num_class  # PTA
    # out = torch.softmax(y_hat, dim=-1)
    # out1 = out.clone().detach() ** gamma
    # label = label * out1
    # loss = cross_entropy1(out, label)
    # loss = loss.sum()
    # loss = torch.sum(weight * loss)
    loss.backward()
    optimizer.step()
    # print(loss.item())
    return loss.item()


def train_altopt_batch(model, data, train_idx, optimizer, args=None):
    # print('train')
    y = data.y
    model.train()
    # torch.autograd.set_detect_anomaly(True) ## to locate error of NaN
    import ipdb; ipdb.set_trace()
    label = model.FF

    if label is None:
        label = model.prop.init_label(data)
    train_mask = data.train_mask
    total_weight = torch.zeros(y.shape[0]).cuda()

    if args.current_epoch == 0:
        num = 0
    else:
        num = 0

    if args.current_epoch > 0:
        for i in range(args.num_class):
            weight = 1 - torch.sum(-label * torch.log(label.clamp(min=1e-8)), 1) / math.log(args.num_class)
            _, index = label.max(dim=1)
            pos = (index == i)
            weight[~pos] = 0
            weight[train_mask] = 0
            value, indices = torch.topk(weight, num)
            total_weight[indices] = value
    total_weight[train_mask] = 1
    # total_weight[:] = 1
    all_train_index = total_weight.nonzero().squeeze()
    # all_train_index =
    batch_size = 1000
    batches = all_train_index.split(batch_size)
    for batch in batches:
        optimizer.zero_grad()
        d = data.x[batch].cuda()
        out = model(data=d)
        pred = out
        FF_label = label[batch]
        diff = cross_entropy1(pred, FF_label)
        loss = torch.mean(total_weight[batch] * diff)
        loss.backward()
        optimizer.step()
        # del d
        # torch.cuda.empty_cache()
    return loss.item()

# def train_altopt(model, data, train_idx, optimizer, args=None):
#     # print('train')
#     y = data.y
#     model.train()
#     # torch.autograd.set_detect_anomaly(True) ## to locate error of NaN
#     optimizer.zero_grad()
#     # if args.current_epoch != 0:
#     #     import ipdb; ipdb.set_trace()
#     out = model(data=data)
#     label = model.FF
#     if label is None:
#         label = model.prop.init_label(data)
#     train_mask = data.train_mask
#     total_weight = torch.zeros(y.shape[0]).cuda()

#     if args.current_epoch == 0:
#         num = 0
#     else:
#         num = 100

#     if args.current_epoch > 0:
#         for i in range(args.num_class):
#             weight = 1 - torch.sum(-label * torch.log(label.clamp(min=1e-8)), 1) / math.log(args.num_class)
#             _, index = label.max(dim=1)
#             pos = (index == i)
#             weight[~pos] = 0
#             weight[train_mask] = 0
#             value, indices = torch.topk(weight, num)
#             total_weight[indices] = 1
            
#     total_weight[train_mask] = 1
#     diff = out - label
#     diff = torch.sum(diff * diff, 1)
#     loss = torch.sum(total_weight * diff)
#     loss.backward()
#     optimizer.step()
#     return loss.item()


def train_altopt(model, data, train_idx, optimizer, args=None):
    # print('train')
    y = data.y
    model.train()
    # torch.autograd.set_detect_anomaly(True) ## to locate error of NaN
    optimizer.zero_grad()
    # if args.current_epoch != 0:
    #     import ipdb; ipdb.set_trace()
    out = model(data=data)
    label = model.FF
    if label is None:
        label = model.prop.init_label(data)
    train_mask = data.train_mask
    total_weight = torch.zeros(y.shape[0]).cuda()

    if args.current_epoch == 0:
        num = 0
    else:
        num = 100
        # num = 50 + args.current_epoch * 10

    if args.current_epoch > 0 and model.total_weight is None:
        for i in range(args.num_class):
            weight = 1 - torch.sum(-label * torch.log(label.clamp(min=1e-8)), 1) / math.log(args.num_class)
            _, index = label.max(dim=1)
            pos = (index == i)
            weight[~pos] = 0
            weight[train_mask] = 0
            value, indices = torch.topk(weight, num)
            total_weight[indices] = 1
        model.total_weight = total_weight
    elif args.current_epoch > 0:
        total_weight = model.total_weight
    total_weight[train_mask] = 1
    # import ipdb; ipdb.set_trace()
    
    # all_train_index = total_weight.nonzero().squeeze()
    # all_train_index = train_mask

    # out = out[all_train_index]
    # label = label[all_train_index]
    # diff = cross_entropy1(out, label)
    # loss = torch.mean(total_weight[all_train_index] * diff)
    # torch.cuda.empty_cache()
    #diff = out - label
    #diff = torch.sum(diff * diff, 1)
    diff = cross_entropy1(out, label)
    loss = torch.sum(total_weight * diff)

    # out = model(data=data)[train_idx]
    
    # if len(data.y.shape) == 1:
    #     y = data.y[train_idx]
    # else:
    #     y = data.y.squeeze(1)[train_idx]  ## for ogb data

    # if args.loss == 'CE':
    #     loss = F.nll_loss(out[train_idx], y)

    # diff = pred - FF_label
    # diff = torch.sum(diff * diff, 1)
    # loss = torch.mean(total_weight[all_train_index] * diff)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_altopt(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    if args.model == 'ALTOPT':
        out = model(data=data)  # still forward to update mlp output, or move it to propagate_update
        # out = torch.softmax(out, dim=-1)
        out = model.FF  #hidden
        # out = model.ensamble(data, out)
    else:
        out = model(data=data)
    
    # out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    # import ipdb; ipdb.set_trace()

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))

    # print(f'train_acc: {train_acc}, valid_acc: {valid_acc}, test_acc: {test_acc}')
    return train_acc, valid_acc, test_acc
    # return -train_loss, -valid_loss, -test_loss
    
def train(model, data, train_idx, optimizer, args=None):
    # print('train')
    model.train()
    # torch.autograd.set_detect_anomaly(True) ## to locate error of NaN
    optimizer.zero_grad()
    out = model(data=data)[train_idx]
    
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data

    if args.loss == 'CE':
        loss = F.nll_loss(out, y)
    elif args.loss == 'MSE':
        # convert y to one-hot format
        label = torch.zeros_like(out) 
        label[range(y.shape[0]), y] = 1
        # import ipdb; ipdb.set_trace()
        loss = torch.pow(torch.norm(out-label), 2)
    # print('####### training loss: ', loss)
    loss.backward()
    # print('loss: ', loss)
    optimizer.step()
    return loss.item()

def train11(model, data, train_idx, optimizer, args=None):
    # print('train')
    model.train()
    # torch.autograd.set_detect_anomaly(True) ## to locate error of NaN
    # import ipdb; ipdb.set_trace()
    optimizer.zero_grad()
    out = model(data=data)[train_idx]
    
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data

    if args.loss == 'CE':
        loss = F.nll_loss(out, y)
    elif args.loss == 'MSE':
        # convert y to one-hot format
        label = torch.zeros_like(out) 
        label[range(y.shape[0]), y] = 1
        # import ipdb; ipdb.set_trace()
        loss = torch.pow(torch.norm(out-label), 2)
    # print('####### training loss: ', loss)
    loss.backward()
    # print('loss: ', loss)
    optimizer.step()
    return loss.item()

def test_top(model, data, split_idx, args=None):
    label = model(data=data)
    train_mask = data.train_mask
    y = data.y
    total_weight = torch.zeros(y.shape[0]).cuda()
    num = 300
    label = label.softmax(dim=-1)
    for i in range(args.num_class):
        weight = 1 - torch.sum(-label * torch.log(label.clamp(min=1e-8)), 1) / math.log(args.num_class)
        # import ipdb; ipdb.set_trace()
        _, index = label.max(dim=1)
        pos = (index == i)
        # test_distribution.append(sum(pos).item())
        weight[~pos] = 0
        weight[train_mask] = 0
        value, indices = torch.topk(weight, num)
        total_weight[indices] = 1

    psuedo_indices = total_weight.nonzero().squeeze()
    true_label = y[psuedo_indices].unsqueeze(dim=1)
    psuedo_label = label.argmax(dim=1, keepdim=True)[psuedo_indices]
    evaluator = Evaluator(name='ogbn-arxiv')
    if len(psuedo_indices) > 0:
        psuedo_acc = evaluator.eval({
            'y_true': true_label,
            'y_pred': psuedo_label,
        })['acc']
        print('psuedo_acc', psuedo_acc)
        
        
@torch.no_grad()
def test(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    # if args.model == 'ALTOPT':
    #     out = model.FF ## hidden
    # else:
    #     out = model(data=data)
    out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))
    
    return train_acc, valid_acc, test_acc, y_pred


@torch.no_grad()
def test_f1(model, data, split_idx, args=None):
    model.eval()
    out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y
    macro_f1 = f1_score(y[split_idx['test']].cpu().numpy(), y_pred[split_idx['test']].cpu().numpy(), average='macro')
    return macro_f1

@torch.no_grad()
def test11(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    # if args.model == 'ALTOPT':
    #     out = model.FF ## hidden
    # else:
    #     out = model(data=data)
    out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))
    
    return train_acc, valid_acc, test_acc
    # return out, train_acc, valid_acc, test_acc
    # return -train_loss, -valid_loss, -test_loss


def train_appnp(model, data, train_idx, optimizer, args=None):
    # print('train')
    model.train()
    torch.autograd.set_detect_anomaly(True)  ## to locate error of NaN
    optimizer.zero_grad()
    out, loss1 = model(data=data)
    out = out[train_idx]
    # out = model(data=data)[train_idx]

    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data

    if args.loss == 'CE':
        if args.current_epoch > 100:
            loss = F.nll_loss(out, y) + 0.005 * loss1
            # loss = F.nll_loss(out, y)
        else:
            loss = F.nll_loss(out, y)
        # loss = F.nll_loss(out, y)
    elif args.loss == 'MSE':
        # convert y to one-hot format
        label = torch.zeros_like(out)
        label[range(y.shape[0]), y] = 1
        # import ipdb; ipdb.set_trace()
        loss = torch.pow(torch.norm(out - label), 2)
    # print('####### training loss: ', loss)
    loss.backward()
    # print('loss: ', loss)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_appnp(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    # if args.model == 'ALTOPT':
    #     out = model.FF ## hidden
    # else:
    #     out = model(data=data)
    out, diff = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))

    return train_acc, valid_acc, test_acc


@torch.no_grad()
def test1(model, data, out, split_idx, args=None):
    # print('test')
    model.eval()
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    print('base model test_acc', test_acc)
    return train_acc, valid_acc, test_acc


def train_cs(model, data, train_idx, optimizer, args=None):
    model.train()
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    out = model(data=data)[train_idx]
    out = F.log_softmax(out, dim=-1)
    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data
    loss = F.nll_loss(out, y)
    # loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test_cs(model, data, split_idx, out=None, args=None):
    model.eval()
    if out is None:
        out = model(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)
    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1) # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    return train_acc, valid_acc, test_acc, out


def train_finetune(model, data, train_idx, optimizer, args=None):
    # print('train')
    model.train()
    # torch.autograd.set_detect_anomaly(True) ## to locate error of NaN
    optimizer.zero_grad()
    out = model.finetune(data=data)[train_idx]

    if len(data.y.shape) == 1:
        y = data.y[train_idx]
    else:
        y = data.y.squeeze(1)[train_idx]  ## for ogb data

    if args.loss == 'CE':
        loss = F.nll_loss(out, y)
    elif args.loss == 'MSE':
        # convert y to one-hot format
        label = torch.zeros_like(out)
        label[range(y.shape[0]), y] = 1
        # import ipdb; ipdb.set_trace()
        loss = torch.pow(torch.norm(out - label), 2)
    # print('####### training loss: ', loss)
    loss.backward()
    # print('loss: ', loss)
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_finetune(model, data, split_idx, args=None):
    # print('test')
    model.eval()
    # if args.model == 'ALTOPT':
    #     out = model.FF ## hidden
    # else:
    #     out = model(data=data)
    out = model.finetune(data=data)
    y_pred = out.argmax(dim=-1, keepdim=True)

    if len(data.y.shape) == 1:
        y = data.y.unsqueeze(dim=1)  # for non ogb datas
    else:
        y = data.y

    evaluator = Evaluator(name='ogbn-arxiv')
    train_acc = evaluator.eval({
        'y_true': y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']
    # y_true_test = y[split_idx['test']]
    # y_pred = y_pred[split_idx['test']]
    # if args.current_epoch % 100 == 0:
    #     for i in range(7):
    #         pos = (y_pred==i)
    #         y_true_i = y_true_test[pos]
    #         y_pred_i = y_pred[pos]
    #         print(i, torch.sum(y_pred_i==y_true_i)/len(y_pred_i))

    return train_acc, valid_acc, test_acc

if __name__ == '__main__':
    print('hh')
