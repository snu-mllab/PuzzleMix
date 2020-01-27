import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gco
from ortools.graph import pywrapgraph
import copy
from lapjv import lapjv

class SimpleHungarianSolver:
    def __init__(self, nworkers, ntasks, value=100000):
        '''
        This can be used when nworkers*k > ntasks
        Args:
            nworkers - int
            ntasks - int
            value - int
                should be large defaults to be 100000
            pairwise_lamb - int
        '''
        self.nworkers = nworkers
        self.ntasks = ntasks
        self.value = value

        self.source = 0
        self.sink = self.nworkers+self.ntasks+1

        self.supplies = [self.nworkers]+(self.ntasks+self.nworkers)*[0]+[-self.nworkers]
        self.start_nodes = list()
        self.end_nodes = list()
        self.capacities = list()
        self.common_costs = list()

        self.start_nodes = np.ndarray([self.nworkers*self.ntasks+self.nworkers+self.ntasks])
        self.end_nodes = np.ndarray([self.nworkers*self.ntasks+self.nworkers+self.ntasks])
        self.capacities = np.ndarray([self.nworkers*self.ntasks+self.nworkers+self.ntasks])
        self.common_costs = np.ndarray([self.nworkers+self.ntasks])


        for work_idx in range(self.nworkers):
            self.start_nodes[work_idx] = self.source
            self.end_nodes[work_idx] = work_idx+1
            self.capacities[work_idx] = 1
            self.common_costs[work_idx] = 0

        for task_idx in range(self.ntasks):
            self.start_nodes[task_idx+nworkers] = nworkers+1+task_idx
            self.end_nodes[task_idx+nworkers] = nworkers+ntasks+1
            self.capacities[task_idx+nworkers] = 1
            self.common_costs[task_idx+nworkers] = 0

        idx = nworkers+ntasks
        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                self.start_nodes[idx] = work_idx+1
                self.end_nodes[idx] = nworkers+1+task_idx
                self.capacities[idx] = 1
                idx += 1

        self.nnodes = len(self.start_nodes)
    
    def solve(self, array):
        assert array.shape == (self.nworkers, self.ntasks), "Wrong array shape, it should be ({}, {})".format(self.nworkers, self.ntasks)
        
        s = time()
        self.array = self.value*array
        self.array = -self.array # potential to cost
        self.array = self.array.astype(np.int32)

        costs = np.ndarray([self.nworkers*self.ntasks+self.nworkers+self.ntasks])
        costs[:len(self.common_costs)] = copy.copy(self.common_costs)
        #costs = copy.copy(self.common_costs)
        idx = len(self.common_costs)
        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                costs[idx] = self.array[work_idx][task_idx]
                idx += 1

        #costs = np.array(costs)
        costs = list(map(int, (costs.tolist())))

        start_nodes = list(map(int, self.start_nodes.tolist()))
        end_nodes = list(map(int, self.end_nodes.tolist()))
        capacities = list(map(int, self.capacities.tolist()))
        common_costs = list(map(int, self.common_costs.tolist()))
        nnodes = self.nnodes

        nworkers = self.nworkers
        ntasks = self.ntasks

        supplies = self.supplies
        source = self.source
        sink = self.sink
        
        assert len(costs)==nnodes, "Length of costs should be {} but {}".format(nnodes, len(costs))
        min_cost_flow = pywrapgraph.SimpleMinCostFlow()
        for idx in range(nnodes):
             min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[idx], end_nodes[idx], capacities[idx], costs[idx])
                
        for idx in range(ntasks+nworkers+2):
            min_cost_flow.SetNodeSupply(idx, supplies[idx])
        insert_time = time() - s

        s = time()
        min_cost_flow.Solve()
        
        results = np.ndarray([nworkers, 2])
        idx = 0
        for arc in range(min_cost_flow.NumArcs()):
            tail = min_cost_flow.Tail(arc)
            head = min_cost_flow.Head(arc)
            if tail != source and head != sink:
                if min_cost_flow.Flow(arc) > 0:
                    results[idx,0] = tail-1
                    results[idx,1] = head-nworkers-1
                    idx+=1
                    #results.append([tail-1, head-nworkers-1])
        solve_time = time() - s

        s = time()
        results_np = np.zeros_like(array)
        for i in range(nworkers): results_np[int(results[i,0])][int(results[i,1])]=1
        convert_time = time() - s

        return results, results_np, insert_time, solve_time, convert_time

def l2_norm(matrix):
    return torch.sqrt(torch.sum(matrix*matrix))

def cosine(unary1, unary2, unary1_l2, unary2_l2):
    return 1 - torch.sum(unary1*unary2) / (unary1_l2 * unary2_l2)


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
    from time import time

    block_num_list= [2, 4, 8, 16]
    print('100 sample test')
    for block_num in block_num_list:
        print("\ngraph size: {}x{}".format(block_num, block_num))
        ### unit test
        n_samples = 100
        np.random.seed(0)
        
        eps = 0.2
        C = cost_matrix(block_num).unsqueeze(0)
        hun_time = 0
        to_cpu_time = 0
        graph_in_time = 0
        graph_solve_time = 0
        graph_plan_time = 0
        to_gpu_time = 0
        our_time = 0

        hungarian_solver = SimpleHungarianSolver(nworkers=block_num**2, ntasks=block_num**2)

        mask = torch.tensor(np.random.randint(0,2 ,size=(n_samples, block_num, block_num)), dtype=torch.float32, device='cuda')
        unary = torch.tensor(np.random.uniform(size=(n_samples, block_num, block_num)), dtype=torch.float32, device='cuda')
        cost = (eps * C - unary.reshape(n_samples, block_num**2, 1) * mask.reshape(n_samples, 1, block_num**2)) * mask.reshape(n_samples, 1, block_num**2)

        s= time()
        plan = mask_transport(mask, unary, eps=eps, n_iter=block_num, t_type = 'full') 
        our_time = time() -s

        plan_indices = []
        s= time()
        for i in range(n_samples):
            plan_indices.append(torch.tensor(lapjv(cost[i].cpu().numpy())[0], dtype=torch.long, device='cuda'))
        plan_indices = torch.stack(plan_indices, dim=0)
        plan_hungarian = torch.zeros_like(cost).scatter_(-1, plan_indices.unsqueeze(-1), 1) 
            
            # _, plan_hungarian, insert_time, solve_time, plan_convert_time = hungarian_solver.solve(cost_np) 
            # graph_in_time += insert_time; graph_solve_time += solve_time; graph_plan_time += plan_convert_time

            # s = time()
            # plan_hungarian = torch.tensor(plan_hungarian, device='cuda')
            # to_gpu_time += time() -s 

        hun_time += time() - s


        for i in range(n_samples):
            if (plan_hungarian[i] * cost[i]).sum() >= (plan[i] * cost[i]).sum() + 1e-5:
                print("Cost by Hungarian: ", (plan_hungarian[i] * cost[i]).sum())
                print("Cost by Ours: ", (plan[i] * cost[i]).sum())
                print(plan_hungarian[i])
                print(plan[i])
                print(cost[i])

                raise AssertionError('Hungarian Error!')

        print('hungarian time : {:.4f}'.format(hun_time))
        # print('- to cpu       : {:.4f}'.format(to_cpu_time))
        # print('- draw graph   : {:.4f}'.format(graph_in_time))
        # print('- solve graph  : {:.4f}'.format(graph_solve_time))
        # print('- convert plan : {:.4f}'.format(graph_plan_time))
        # print('- to gpu       : {:.4f}'.format(to_gpu_time))
        print('our time       : {:.4f}'.format(our_time))

