import numpy as np
import torch
import itertools


def mask_transport(mask, grad_pool, eps=0.01, n_iter=5, t_type='full'):
    if n_iter is None:
        n_iter = int(np.sqrt(width))

    batch_size = mask.shape[0]
    width = mask.shape[-1]
    C = cost_matrix(width).unsqueeze(0)    
    if t_type =='full':
        z = (mask>0).float()
    else:
        z = mask
    
    cost = eps * C - grad_pool.reshape(-1, block_num**2, 1) * z.reshape(-1, 1, block_num**2)
    
    # init
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
    block_num = 3
    n_sample = 1000
    eps= 0.2
    C = cost_matrix(block_num).unsqueeze(0)

    plan_elm= [row for row in np.eye(block_num**2)]
    plan_list = [torch.tensor(plan, dtype=torch.float32) for plan in list(itertools.permutations(plan_elm))]


    print('transport algorithm test for {} block_num, {} samples'.format(block_num, n_sample))
    incorrect = 0
    incorrect_elm = 0
    positive = 0
    for i in range(n_sample):
        mask = torch.tensor(np.random.randint(0,2 ,size=(block_num, block_num)), dtype=torch.float32)
        unary = torch.tensor(np.random.uniform(size=(block_num, block_num)), dtype=torch.float32)

        cost = eps * C * mask.reshape(-1, 1, block_num**2) - unary.reshape(-1, block_num**2, 1) * mask.reshape(-1, 1, block_num**2)

        ### Greedy Search
        ## Note there can be multiple global optima
        plan_best_list = []
        obj_best = 0
        for plan in plan_list:
            obj = torch.sum(plan * cost)
            if obj < obj_best:
                obj_best = obj
                plan_best_list = [plan * mask.reshape(-1, 1, block_num**2)]
            elif obj == obj_best:
                duplicate = 0            
                for plan_best in plan_best_list:
                    if torch.sum(plan_best != (plan * mask.reshape(-1, 1, block_num**2))) == 0:
                        duplicate = 1
                        break
                if not duplicate:
                    plan_best_list.append(plan * mask.reshape(-1, 1, block_num**2))

        ### By our algorithm
        plan = mask_transport(mask, unary, eps=eps, n_iter=block_num**2)  * mask.reshape(-1, 1, block_num**2)

        dist = block_num
        for plan_best in plan_best_list:
            dist = min(dist, int(torch.sum(plan_best != plan)))
            
        positive += int(mask.sum())
        incorrect_elm += dist
        if dist > 0:
            incorrect += 1
        
        if (i+1)%10==0:
            print('(sample ) {}/{}'.format(incorrect, i+1))
            print('(element) {}/{}'.format(incorrect_elm, positive))

    print('suboptimal (sample): {}/{}'.format(incorrect, n_sample))
    print('suboptimal (element): {}/{}'.format(incorrect_elm, positive))

