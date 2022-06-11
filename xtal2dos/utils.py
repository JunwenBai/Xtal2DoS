import numpy as np
from scipy import stats
import torch
import pandas as pd
import os
import shutil
import argparse
from operator import attrgetter

from   sklearn.model_selection import train_test_split
from   sklearn.metrics import mean_absolute_error as sk_MAE
from   tabulate import tabulate
import random,time

from xtal2dos.pytorch_stats_loss import torch_wasserstein_loss
from sklearn.metrics import r2_score, mean_squared_error


def comp_wd(x1, x2):
    #x1 = x1+1e-6
    #x2 = x2+1e-6
    bs = x1.shape[0]
    dim = x1.shape[1]
    vec_list = np.arange(dim)
    #x1 = x1 / np.sum(x1, axis=-1, keepdims=True)
    #x2 = x2 / np.sum(x2, axis=-1, keepdims=True)

    assert np.isfinite(x1).all()
    assert np.isfinite(x2).all()

    # Making Scipy Results
    result_1=0
    for i in range(bs):
        vec_dist_1 = stats.wasserstein_distance(vec_list, vec_list, x1[i], x2[i])
        result_1 += vec_dist_1
    return result_1*0.0625/bs

def kl_divergence(p, q):
    return np.mean(np.sum(p * np.log(p/(q+1e-8)+1e-8), axis=-1))

def comp_kl(x1, x2):
    #x1 = x1+1e-6
    #x2 = x2+1e-6
    x1 = x1 / np.sum(x1, axis=-1, keepdims=True)
    x2 = x2 / np.sum(x2, axis=-1, keepdims=True)
    assert (x1 >= 0).all()
    assert (x2 >= 0).all()
    return np.mean(np.sum(kl_divergence(x1, x2), axis=-1))

def r2(x1, x2):
    return r2_score(x1, x2, multioutput='variance_weighted')


def metrics(label_gt, prediction, args, mode='normalized_sum'):
    if mode == 'standardized':
        mean = NORMALIZER.mean.detach().numpy()
        std = NORMALIZER.std.detach().numpy()
        label_gt_standardized = (label_gt-mean)/std
        prediction = (prediction-mean)/std
        #mae = np.mean(np.abs((prediction)-label_gt_standardized))
        #mae_x = np.mean(np.abs((prediction_x)-label_gt_standardized))
        mae, r2, mse, wd = eval_metrics(prediction, label_gt_standardized)

        prediction = prediction*std+mean
        prediction[prediction < 0] = 0
        #mae_ori = np.mean(np.abs((prediction)-label_gt))
        #mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))
        mae_ori, r2_ori, mse_ori, wd_ori = eval_metrics(prediction, label_gt)

    elif mode == 'normalized_max':
        label_max = np.expand_dims(np.max(label_gt, axis=1), axis=1)
        label_gt_standardized = label_gt / label_max

        pred_max = np.expand_dims(np.max(prediction, axis=1), axis=1)
        prediction = prediction / pred_max
        #mae = np.mean(np.abs((prediction)-label_gt_standardized))
        #mae_x = np.mean(np.abs((prediction_x)-label_gt_standardized))
        mae, r2, mse, wd = eval_metrics(prediction, label_gt_standardized)

        prediction = prediction*label_max
        #mae_ori = np.mean(np.abs((prediction)-label_gt))
        #mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))
        mae_ori, r2_ori, mse_ori, wd_ori = eval_metrics(prediction, label_gt)

    elif mode == 'normalized_sum':
        label_sum = np.sum(label_gt, axis=1, keepdims=True)
        label_gt_standardized = label_gt / label_sum

        pred_sum = np.sum(prediction, axis=1, keepdims=True)
        prediction = prediction / pred_sum
        #mae = np.mean(np.abs((prediction)-label_gt_standardized))
        #mae_x = np.mean(np.abs((prediction_x)-label_gt_standardized))
        mae, r2, mse, wd = eval_metrics(prediction, label_gt_standardized)

        prediction = prediction*label_sum
        #mae_ori = np.mean(np.abs((prediction)-label_gt))
        #mae_x_ori = np.mean(np.abs((prediction_x)-label_gt))
        mae_ori, r2_ori, mse_ori, wd_ori = eval_metrics(prediction, label_gt)

    return mae, r2, mse, wd, mae_ori, r2_ori, mse_ori, wd_ori


def eval_metrics(pred, label):
    mae = np.mean(np.abs((pred) - label))
    r2 = r2_score(label, pred, multioutput='variance_weighted')
    mse = mean_squared_error(pred, label)
    wd = comp_wd(pred, label) 
    return mae, r2, mse, wd


#def set_device(gpu_id=0):
#    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
#    return device

def set_model_properties(crystal_property):
    if crystal_property   in ['poisson-ratio','band-gap','absolute-energy','fermi-energy','formation-energy']:
        norm_action   = None; classification = None
    elif crystal_property == 'is_metal':
        norm_action   = 'classification-1'; classification = 1
    elif crystal_property == 'is_not_metal':
        norm_action   = 'classification-0'; classification = 1
    else:    
        norm_action   = 'log'; classification = None
    return norm_action, classification

def torch_MAE(tensor1,tensor2):
    return torch.mean(torch.abs(tensor1-tensor2))

def torch_accuracy(pred_tensor,true_tensor):
    _,pred_tensor   = torch.max(pred_tensor,dim=1)
    correct         = (pred_tensor==true_tensor).sum().float()
    total           = pred_tensor.size(0)
    accuracy_ans    = correct/total
    return accuracy_ans

def output_training(metrics_obj,epoch,estop_val,extra='---'):
    header_1, header_2 = 'MSE | e-stop','MAE | TIME'
    if metrics_obj.c_property in ['is_metal','is_not_metal']:
        header_1,header_2     = 'Cross_E | e-stop','Accuracy | TIME'

    train_1,train_2 = metrics_obj.training_loss1[epoch],metrics_obj.training_loss2[epoch]
    valid_1,valid_2 = metrics_obj.valid_loss1[epoch],metrics_obj.valid_loss2[epoch]
    
    tab_val = [['TRAINING',f'{train_1:.4f}',f'{train_2:.4f}'],['VALIDATION',f'{valid_1:.4f}',f'{valid_2:.4f}'],['E-STOPPING',f'{estop_val}',f'{extra}']]
    
    output = tabulate(tab_val,headers= [f'EPOCH # {epoch}',header_1,header_2],tablefmt='fancy_grid')
    print(output)
    return output

def load_metrics():
    saved_metrics = pickle.load(open("MODELS/metrics_.pickle", "rb", -1))
    return saved_metrics


def freeze_params(model, params_to_freeze_list):
    for str in params_to_freeze_list:
        attr = attrgetter(str)(model)
        attr.requires_grad = False
        attr.grad = None


def unfreeze_params(model, params_to_unfreeze_list):
    for str in params_to_unfreeze_list:
        attr = attrgetter(str)(model)
        #print(str)
        #print(attr)
        attr.requires_grad = True


def RobustL1(output, log_std, target):
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    absolute = torch.abs(output - target)
    loss = np.sqrt(2.0) * absolute * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2(output, log_std, target):
    """
    Robust L2 loss using a gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    squared = torch.pow(output - target, 2.0)
    loss = 0.5 * squared * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)
