import math
import numpy as np
import gco



"""test Code for multi-label graph cut"""
def test_graph(unary1, unary2, pw_x, pw_y, beta, n_labels=2, verbose=False):
    block_num = unary1.shape[-1]
    input1 = np.copy(unary1)
    input2 = np.copy(unary2)
    
    # Find Optimal labeling by exhaustive search
    mask_true = np.zeros_like(unary1)
    loss_min = 10000
    
    for num in range(n_labels**(block_num**2)):
        mask =[]
        for _ in range(block_num**2):
            mask.append(float(num%n_labels)/(n_labels-1))
            num = num//n_labels        
    
        mask = np.array(mask).reshape(block_num, block_num)
        loss = np.sum((1-mask) * unary1) + np.sum(mask* unary2) +\
                  np.sum(pw_x[1,1] * mask[:-1, :] * mask[1:, :]) + np.sum((pw_x[0,1] + beta) * (1 - mask[:-1, :]) * mask[1:, :]) +\
                  np.sum((pw_x[1,0] + beta) * mask[:-1, :] * (1-mask[1:, :])) + np.sum(pw_x[0,0] * (1 - mask[:-1, :])* (1-mask[1:, :])) +\
                  np.sum(pw_y[1,1] * mask[:, :-1] * mask[:, 1:]) + np.sum((pw_y[0,1] + beta) * (1 - mask[:, :-1]) * mask[:, 1:]) +\
                  np.sum((pw_y[1,0] + beta) * mask[:, :-1] * (1-mask[:, 1:])) + np.sum(pw_y[0,0] * (1 - mask[:, :-1])* (1-mask[:, 1:]))
        
        if loss < loss_min:
            loss_min = loss
            mask_true = np.copy(mask)
            
            
    # Find Optimal labeling by graph cut
    unary2[:-1,:] += (pw_x[1,0] + pw_x[1,1])/2.
    unary1[:-1,:] += (pw_x[0,1] + pw_x[0,0])/2.
    unary2[1:,:] += (pw_x[0,1] + pw_x[1,1])/2.
    unary1[1:,:] += (pw_x[1,0] + pw_x[0,0])/2.

    unary2[:,:-1] += (pw_y[1,0] + pw_y[1,1])/2
    unary1[:,:-1] += (pw_y[0,1] + pw_y[0,0]) / 2
    unary2[:,1:] += (pw_y[0,1] + pw_y[1,1])/2
    unary1[:,1:] += (pw_y[1,0] + pw_y[0,0]) / 2
    pw_x = (pw_x[1,0] + pw_x[0,1] - pw_x[1,1] - pw_x[0,0])/2
    pw_y = (pw_y[1,0] + pw_y[0,1] - pw_y[1,1] - pw_y[0,0])/2
    
    mask_graph = graphcut_multi(unary2, unary1, pw_x, pw_y, beta, n_labels=n_labels)
    
    if verbose:
        print("true mask: \n", mask_true)
        print("graph_cut mask: \n", mask_graph) 

    if np.sum(mask_graph != mask_true)>0:
        print("true mask: \n", mask_true)
        print("graph_cut mask: \n", mask_graph)
        print("inputs: \n", input1, '\n', input2)
        print("pw_x: \n", pw_x)
        print("beta: ", beta)
        raise AssertionError("Different Results!")


def graphcut_multi(unary1, unary2, pw_x, pw_y, beta, n_labels=2):
    block_num = unary1.shape[0]
    alpha = 1000 * block_num ** 2 
    unary_cost =  np.stack([(1-i) * alpha * unary1 + i * alpha * unary2 for i in np.linspace(0,1, n_labels)], axis=-1).astype(np.int32)

    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32)
    for i in range(n_labels):
        for j in range(n_labels):
            pairwise_cost[i, j] = (1-i/2)*j/2 + i/2*(1-j/2)


    pw_x = (alpha * (pw_x + beta)).astype(np.int32)
    pw_y = (alpha * (pw_y + beta)).astype(np.int32)
    
    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y, algorithm='swap')/(n_labels-1)
    mask = labels.reshape(block_num, block_num)
    
    return mask
    
    
if __name__ == '__main__':
    block_num = 3
    n_labels = 3
    print('one sample test')
    beta=0.1

    np.random.seed(0)
    unary_test1 = np.random.uniform(size=(block_num,block_num))
    unary_test2 = np.random.uniform(size=(block_num,block_num))
    pw_x = np.random.uniform(size=(2,2,block_num-1,block_num)) * beta
    pw_y = np.random.uniform(size=(2,2,block_num,block_num-1)) * beta

    test_graph(unary_test1, unary_test2, pw_x, pw_y, beta=beta, n_labels=n_labels, verbose=True)

    print('\n100 samples test (If different, error will occur. For n_labels=3 it will take few minutes.)')
    for i in range(100):
        unary_test1 = np.random.uniform(size=(block_num,block_num))
        unary_test2 = np.random.uniform(size=(block_num,block_num))
        pw_x = np.random.uniform(size=(2,2,block_num-1,block_num)) * beta
        pw_y = np.random.uniform(size=(2,2,block_num,block_num-1)) * beta

        test_graph(unary_test1, unary_test2, pw_x, pw_y, beta=beta, n_labels=n_labels)
    print("test finished")

