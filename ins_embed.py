import numpy as np
import torch
import torch.nn as nn


def calculate_means(pred, gt, n_objects, max_n_objects, usegpu):
    '''
    Operation:
    Calculate the mean of each instance embedding
    ---------------------------------------------------------------
    pred: bs, num_point, dim
    gt: bs, num_point, n_instances
    n_objects: bs, 1
    max_n_objects: int
    '''
    bs, n_loc, n_filters = pred.shape
    n_instances = gt.shape[2]

    pred_repeated = pred.unsqueeze(2).expand(
        bs, n_loc, n_instances, n_filters)  # bs, num_point, n_instances, n_filters
    gt_expanded = gt.unsqueeze(3) # bs, num_point, n_instances, 1
    # instance mask 
    pred_masked = pred_repeated * gt_expanded

    means = []
    for i in range(bs):
        _n_objects_sample = n_objects[i]
        # num_point, n_objects, n_filters
        _pred_masked_sample = pred_masked[i, :, : _n_objects_sample]
        # num_point, n_objects, 1
        _gt_expanded_sample = gt_expanded[i, :, : _n_objects_sample]

        _mean_sample = _pred_masked_sample.sum(0) / _gt_expanded_sample.sum(0)  # n_objects, n_filters
        #if (max_n_objects - _n_objects_sample) >= 0: # fill the blank instance
        if False:
            n_fill_objects = max_n_objects - _n_objects_sample
            _fill_sample = torch.zeros(n_fill_objects, n_filters)
            if usegpu:
                _fill_sample = _fill_sample.cuda()
            #_fill_sample = Variable(_fill_sample)
            _mean_sample = torch.cat((_mean_sample, _fill_sample), dim=0)
        means.append(_mean_sample)

    means = torch.stack(means) 

    # means = pred_masked.sum(1) / gt_expanded.sum(1)
    # # bs, n_instances, n_filters

    return means

def calculate_variance_term(pred, gt, means, n_objects, delta_v, norm=2):
    '''
    Operation:
    Calculate the intra-instance points distance
    ---------------------------------------------------------------
    pred: bs, num_point, n_filters
    gt: bs, num_point, n_instances
    means: bs, n_instances, n_filters
    '''
    bs, n_loc, n_filters = pred.size()
    n_instances = gt.size(2)

    # bs, num_point, n_instances, n_filters
    means = means.unsqueeze(1).expand(bs, n_loc, n_instances, n_filters)
    # bs, num_point, n_instances, n_filters
    pred = pred.unsqueeze(2).expand(bs, n_loc, n_instances, n_filters)
    # bs, num_point, n_instances, n_filters
    gt = gt.unsqueeze(3).expand(bs, n_loc, n_instances, n_filters)

    _var = (torch.clamp(torch.norm((pred - means), norm, 3) - delta_v, min=0.0)**2) * gt[:, :, :, 0]

    var_term = 0.0
    for i in range(bs):
        _var_sample = _var[i, :, :n_objects[i]]  # n_loc, n_objects
        _gt_sample = gt[i, :, :n_objects[i], 0]  # n_loc, n_objects

        var_term += torch.sum(_var_sample) / torch.sum(_gt_sample)
    var_term = var_term / bs

    return var_term

def calculate_distance_term(means, n_objects, delta_d, norm=2, usegpu=True):
    '''
    Operation:
    Calculate the inter-instance distance(mean) 
    ---------------------------------------------------------------
    means: bs, n_instances, n_filters
    '''
    bs, n_instances, n_filters = means.size()

    dist_term = 0.0
    no_multiple_ins = False
    for i in range(bs):
        _n_objects_sample = n_objects[i]

        if _n_objects_sample <= 1: # skip if only one instance
            no_multiple_ins = True
            continue

        _mean_sample = means[i, : _n_objects_sample, :]  # n_objects, n_filters
        means_1 = _mean_sample.unsqueeze(1).expand(
            _n_objects_sample, _n_objects_sample, n_filters)
        means_2 = means_1.permute(1, 0, 2)

        diff = means_1 - means_2  # n_objects, n_objects, n_filters
        _norm = torch.norm(diff, norm, 2)

        margin = 2 * delta_d * (1.0 - torch.eye(_n_objects_sample))
        if usegpu:
            margin = margin.cuda()
        #margin = Variable(margin)

        _dist_term_sample = torch.sum(
            torch.clamp(margin - _norm, min=0.0) ** 2)
        _dist_term_sample = _dist_term_sample / (_n_objects_sample * (_n_objects_sample - 1))
        dist_term += _dist_term_sample

    dist_term = dist_term / bs
    if no_multiple_ins:
        dist_term = torch.Tensor([0.0]).cuda()

    return dist_term

def calculate_regularization_term(means, n_objects, norm):
    '''
    Operation:
    Encourage the instance mean toward 0 (avoid unbounded embbeding) 
    -----------------------------------------------------------------
    means: bs, n_instances, n_filters
    '''
    bs, n_instances, n_filters = means.size()

    reg_term = 0.0
    for i in range(bs):
        _mean_sample = means[i, : n_objects[i], :]  # n_objects, n_filters
        _norm = torch.norm(_mean_sample, norm, 1)
        reg_term += torch.mean(_norm)
    reg_term = reg_term / bs

    return reg_term


def discriminative_loss(input, target, n_objects,
                        max_n_objects, delta_v, delta_d, norm, usegpu=True):
    '''
    input: batchsize, num_point, dim
    target: bs, num_point, n_instances
    n_objects: instance number(given by label or prediction), bs
    max_n_objects: max obj exist in scene, bs
    delta_v: distance threshold of var., float
    delta_d: distance threshold of dist., float
    norm: the order of norm when calculate embedding distance, int(1 or 2)
    '''
    # loss coefficients
    alpha = beta = 1.0
    gamma = 0.001

    bs, num_point, n_filters = input.size()
    n_instances = target.size(1)

    #input = input.permute(0, 2, 3, 1).contiguous().view(
    #    bs, height * width, n_filters)
    #target = target.permute(0, 2, 3, 1).contiguous().view(
    #    bs, height * width, n_instances)
    #print(target.shape)
    cluster_means = calculate_means(
        input, target, n_objects, max_n_objects, usegpu)

    var_term = calculate_variance_term(
        input, target, cluster_means, n_objects, delta_v, norm)
    dist_term = calculate_distance_term(
        cluster_means, n_objects, delta_d, norm, usegpu)
    reg_term = calculate_regularization_term(cluster_means, n_objects, norm)
    
    '''
    if torch.isnan(var_term).any():
        print('var_term is Nan...')
    if torch.isnan(dist_term).any():
        print('dist_term is Nan...')
    if torch.isnan(reg_term).any():
        print('reg_term is Nan...')
    '''
    
    loss = alpha * var_term + beta * dist_term + gamma * reg_term
    return loss

class Ins_embed(nn.Module):
    
    def __init__(self, args):
        super(Ins_embed, self).__init__()

    def forward(self, inp):
    # Input: thisbatchsize, self.args.num_point, self.emb_dims*2
        pass

