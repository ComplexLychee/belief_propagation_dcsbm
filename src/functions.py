import numpy as np
from scipy import stats
import time
import graph_tool.all as gt

from dcppm_state import EM_DCPPM
    
def get_degree_prop_zipf(nr, B, gamma, x_max):
    """
    Input
    -----
    nr: number of nodes in each community
    
    B:  number of communities
    
    gamma: shape parameter of the truncated Zipf's distribution
    
    xmamx: a cut-off value of the Zipf's distributon
    
    Returns
    -------
    k_prop: 
    """
    if gamma > 0:
        x = np.arange(1, np.ceil(x_max)+1)
        weights = x ** (-gamma)
        weights /= weights.sum()
        bounded_zipf = stats.rv_discrete(name='bounded_zipf', values=(x, weights))
        degree_prop = bounded_zipf.rvs(size=nr)
        degree_prop = degree_prop/degree_prop.sum()
        degree_prop  = np.concatenate([degree_prop for r in range(B)])
    else:
        degree_prop = np.ones(nr * B)/nr
         
    return degree_prop

def gen_dcppm(nr, B, k, ep, gamma, xmax = None):
    """
    Input
    -----
    nr: number of nodes in each community
    
    B:  number of communities
    
    k : average degree of the network
    
    ep: the strength of the assortative structure, a real value in [0,1]
    
    gamma: shape parameter of the truncated Zipf's distribution
    
    xmamx: a cut-off value of the Zipf's distributon
    
    Returns
    -------
    g: a graph-tool network object
    
    ers: the connection matrix consistting of the number of connections between group r and s
    
    b: the planted network partition
    
    degree_prop: the degree propensity parameter used to generate the network
    
    """
    
    if gamma != 0 and xmax is None:
        raise "Need to specify the cut-off value of the truncated Zipf's distribution"
    
    N = nr*B
    
    p_in = (1+(B-1)*ep)*k/N
    p_out= (1 - ep)*k/N

    ers = np.zeros((B,B))
    for r in range(B):
        for s in range(B):
            if r == s:
                ers[r,s] = nr*nr*p_in
            else:
                ers[r,s] = nr*nr*p_out
    
    b = []
    for r in range(B):
        b += [r for u in range(nr)]
        
    degree_prop = get_degree_prop_zipf(nr, B, gamma, xmax)
    g = gt.generate_sbm(b, ers, out_degs = degree_prop)
    gt.remove_parallel_edges(g)
    gt.remove_self_loops(g)

    g.vp.b = g.new_vp("int", b)
    
    return g, ers, b, degree_prop

def compare_bp_running_time(nr, B, k, ep, gamma,xmax):
    """
    Compare the running time of the belief propagation with different updating scheme
    """

    g, ers, b, degree_prop = gen_dcppm(nr, B, k, ep, gamma, xmax)


    state = EM_DCPPM(g, B = B, t = degree_prop, lrs = ers)

    tic_original = time.time()
    state.bp_iter(max_niter = 1, is_randomised=False)
    toc_original = time.time()
    
    tic_fast = time.time()
    state.bp_iter_fast(max_niter = 1, is_randomised=False)
    toc_fast = time.time()
    
    print(f"One iteration of BP with the original update scheme takes: {toc_original - tic_original} secs!")
    print(f"One iteration of BP with the modified update scheme takes: {toc_fast - tic_fast} secs!")
    
    
def check_bp_accuracy(nr, B, k, ep, gamma, xmax, is_modified = True, max_niter = 1e3):
    
    g, ers, b, degree_prop = gen_dcppm(nr, B, k, ep, gamma, xmax)
    
    state = EM_DCPPM(g, B, t = degree_prop, lrs = ers)
    
    b_init = []
    for u in g.vertices():
        b_init.append(np.argmax(state.vm[u]))
    overlap_init = gt.partition_overlap(b,b_init)

    print(f"Is modified: {is_modified}; Partition overlap before BP: {overlap_init}")
    
    if is_modified:
        state.bp_iter_fast(max_niter = max_niter)
    else:
        state.bp_iter(max_niter = max_niter)
        
    b_est = []
    for u in g.vertices():
        b_est.append(np.argmax(state.vm[u]))
    overlap_inferred = gt.partition_overlap(b,b_est)
    print(f"Is modified: {is_modified}; Partition overlap after BP: {overlap_inferred}")
