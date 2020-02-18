import numpy as np
import torch
import itertools
from lapjv import lapjv

def mask_transport(mask, grad_pool, eps=0.01, n_iter=5, t_type='full'):
    batch_size = mask.shape[0]
    width = mask.shape[-1]

    if n_iter is None:
        n_iter = int(width)

    C = cost_matrix(width).unsqueeze(0)    
    if t_type =='half':
        z = mask
    else :
        z = (mask>0).float()

    
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)
    
    
    if t_type=='full' or t_type=='half':
        for _ in range(n_iter):
            # row resolve, with tie-breaking
            row_best = (cost - 1e-10 * cost.min(-2, keepdim=True)[0]).min(-1)[1]
            plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

            # column resolve
            cost_fight = plan * cost
            col_best = cost_fight.min(-2)[1]
            plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
            plan_lose = (1-plan_win) * plan

            # Note unary and mask are less than 1
            cost += plan_lose
 
    elif t_type=='exact':
        plan_indices = []
        cost_np = cost.cpu().numpy()
        for i in range(batch_size):
            plan_indices.append(torch.tensor(lapjv(cost_np[i])[0], dtype=torch.long))
        plan_indices = torch.stack(plan_indices, dim=0)
        plan_win = torch.zeros_like(cost).scatter_(-1, plan_indices.unsqueeze(-1), 1)
 
    return plan_win


def cost_matrix(width=16, dist=2):
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
    C = torch.tensor(C)
    return C


if __name__=='__main__':
    block_num = 2
    n_sample = 1000
    eps= 0.5
    C = cost_matrix(block_num).unsqueeze(0)

    plan_elm= [row for row in np.eye(block_num**2)]
    plan_list = [torch.tensor(plan, dtype=torch.float32).unsqueeze(0) for plan in list(itertools.permutations(plan_elm))]

    np.random.seed(0)
    print('transport algorithm test for {}x{} plan, {} samples'.format(block_num**2, block_num**2, n_sample))
    incorrect = 0
    for i in range(n_sample):
        mask = torch.tensor(np.random.randint(0,2 ,size=(block_num, block_num)), dtype=torch.float32).unsqueeze(0)
        unary = torch.tensor(np.random.uniform(size=(block_num, block_num)), dtype=torch.float32).unsqueeze(0)
        cost = eps * C - unary.reshape(-1, block_num**2, 1) * mask.reshape(-1, 1, block_num**2)

        ### Brute-Force Search
        ## Note there can be multiple global optima
        plan_best_list = []
        obj_best = 0
        for plan in plan_list:
            obj = torch.sum(plan * cost)
            if obj < obj_best:
                obj_best = obj
                plan_best_list = [plan]
            elif obj == obj_best:
                plan_best_list.append(plan)

        ### By exact algorithm
        cost_np = cost.cpu().numpy()
        plan_indices = torch.tensor(lapjv(cost_np[0])[0], dtype=torch.long).unsqueeze(0)
        plan_exact = torch.zeros_like(cost).scatter_(-1, plan_indices.unsqueeze(-1), 1)

        dist = block_num
        for plan_best in plan_best_list:
            dist = min(dist, int(torch.sum(plan_best != plan_exact)))
            
        if dist > 0:
            incorrect += 1
        
        if (i+1)%100==0:
            print('(incorrect/sample): {}/{}'.format(incorrect, i+1))


