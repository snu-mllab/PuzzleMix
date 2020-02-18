import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gco
import copy
from lapjv import lapjv
import itertools
from time import time


def cost_matrix(width=16, dist=2, device='cuda'):
    size = width**2
    C=np.zeros([size, size], dtype=np.float32)
    for m_i in range(size):
        i1 = m_i // width
        j1 = m_i % width
        for m_j in range(size):
            i2 = m_j // width
            j2 = m_j % width
            C[m_i,m_j]= abs(i1-i2)**2 + abs(j1-j2)**2

    C = C/(width-1)**2
    C = torch.tensor(C, device='cuda')
    return C


def mask_transport(mask, grad_pool, eps=0.01, n_iter=None, t_type='full', verbose=False):
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]
    if n_iter is None:
        n_iter = int(block_num)

    C = cost_matrix(block_num).unsqueeze(0)
    
    if t_type =='full':
        z = (mask>0).float()
    else:
        z = mask

    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)
    
    # init
    for step in range(n_iter):
        # row resolve, with tie-breaking
        row_best = (cost - 1e-10 * cost.min(-2, keepdim=True)[0]).min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost + (1 - plan)
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1-plan_win) * plan

        # Note unary and mask are less than 1
        cost += plan_lose
        if verbose:
            print("(step {}) obj: {:.4f}".format(step, torch.sum((plan_win * z.reshape(-1, 1, block_num**2)) * cost)))
        
        if step > 0 and (plan != plan_prev).sum() == 0:
                break
        plan_prev = plan
    
    return plan_win


def transport_image(img, plan, batch_size, block_num, block_size):
    channel= img.shape[1]
    input_patch = img.reshape([batch_size, channel, block_num, block_size, block_num*block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, channel, block_num, block_num, block_size, block_size]).transpose(-2,-1)
    input_patch = input_patch.reshape([batch_size, channel, block_num**2, block_size, block_size]).permute(0,1,3,4,2).unsqueeze(-1)

    input_transport = plan.transpose(-2,-1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(-1).permute(0,1,4,2,3)
    input_transport = input_transport.reshape([batch_size, channel, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, channel, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2,-1).reshape([batch_size, channel, block_num * block_size, block_num * block_size])
    
    return input_transport


def transport_image_loop(img, plan, batch_size, block_num, block_size):
    plan_move = plan.max(-2)[1] * plan.sum(-2)
    plan_move_idx = torch.nonzero(plan_move)

    input_transport = img.clone()
    for idx in plan_move_idx:
        ex_idx = idx[0]
        target_idx = idx[1]
        source_idx = plan_move[ex_idx, target_idx]

        target_idx = [target_idx//block_num, target_idx%block_num]
        source_idx = [source_idx//block_num, source_idx%block_num]

        input_transport[ex_idx,:, target_idx[0]*block_size: (target_idx[0]+1)*block_size, target_idx[1]*block_size: (target_idx[1]+1)*block_size] =\
        img[ex_idx,:, source_idx[0]*block_size: (source_idx[0]+1)*block_size, source_idx[1]*block_size: (source_idx[1]+1)*block_size]

    return input_transport


if __name__ == '__main__':
    block_num_list= [2, 4, 8, 16, 32]
    n_samples = 100
    print('{} sample test'.format(n_samples))
    for block_num in block_num_list:
        print("\ngraph size: {}x{}".format(block_num, block_num))
        
        np.random.seed(0)
        eps = 0.2
        C = cost_matrix(block_num).unsqueeze(0)
        hun_time = 0
        our_time = 0

        # Random Samples 
        mask = torch.tensor(np.random.randint(0,2 ,size=(n_samples, block_num, block_num)), dtype=torch.float32, device='cuda')
        unary = torch.tensor(np.random.uniform(size=(n_samples, block_num, block_num)), dtype=torch.float32, device='cuda')
        unary = unary / unary.view(n_samples, -1).sum(1).view(n_samples, 1, 1)
        cost = (eps * C - unary.reshape(n_samples, block_num**2, 1) * mask.reshape(n_samples, 1, block_num**2)) * mask.reshape(n_samples, 1, block_num**2)

        # Our Algorithm
        s= time()
        plan = mask_transport(mask, unary, eps=eps, n_iter=block_num, t_type = 'full') 
        our_time = time() -s

        # Exact Algorithm
        plan_indices = []
        s= time()
        for i in range(n_samples):
            plan_indices.append(torch.tensor(lapjv(cost[i].cpu().numpy())[0], dtype=torch.long, device='cuda'))
        plan_indices = torch.stack(plan_indices, dim=0)
        plan_exact = torch.zeros_like(cost).scatter_(-1, plan_indices.unsqueeze(-1), 1) 
        hun_time += time() - s
       
        rel_err = 0
        random_obj_total = 0
        heuristic_obj_total = 0
        exact_obj_total = 0

        # Check err, time
        plan_id = torch.eye(block_num**2, device='cuda')
        for i in range(n_samples):
            # Random Plan
            plan_random = plan_id[torch.randperm(plan_id.shape[0])]
            
            random_obj = (plan_random * cost[i]).sum()
            heuristic_obj = (plan[i] * cost[i]).sum()
            exact_obj = (plan_exact[i] * cost[i]).sum()

            random_err =  random_obj - exact_obj
            heuristic_err =  heuristic_obj - exact_obj

            random_obj_total += random_obj
            heuristic_obj_total += heuristic_obj
            exact_obj_total += exact_obj

            rel_err += heuristic_err / (random_err + heuristic_err + 1e-8)

            # Compare Exact and Heuristic Algorithms
            if exact_obj >= heuristic_obj + 1e-5:
                print("Cost by Hungarian: ", (plan_hungarian[i] * cost[i]).sum())
                print("Cost by Ours: ", (plan[i] * cost[i]).sum())
                print(plan_hungarian[i])
                print(plan[i])
                print(cost[i])

                raise AssertionError('Hungarian Error!')

        print('relative error rate : {:7.4f}'.format(rel_err/n_samples))
        print('random objective    : {:7.4f}'.format(random_obj_total/n_samples))
        print('heuristic objective : {:7.4f}'.format(heuristic_obj_total/n_samples))
        print('exact objective     : {:7.4f}'.format(exact_obj_total/n_samples))

        print('hungarian time : {:.4f}'.format(hun_time))
        print('our time       : {:.4f}'.format(our_time))

