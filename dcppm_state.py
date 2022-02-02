import graph_tool.all as gt
import numpy as np
from scipy import math
import random
lgamma = math.lgamma
log = np.log
exp = np.exp
import graph_tool.all as gt

def normalise(v):
    """
    Normalise a vector such that the exponetial of the verctor is a probability distribution
    """
    
    max_v = max(v)
    log_sum = 0
    for x in v:
        log_sum += exp(x - max_v)
    log_sum = log(log_sum) + max_v
    
    for idx, x in enumerate(v):
        v[idx] = exp(x - log_sum)
        
def h(t1, t2, lrs, Aij):
    
    if lrs > 0:
        log_h = -t1*t2*lrs + Aij*log(t1*t2*lrs) - lgamma(Aij + 1)
    
        return np.exp(log_h)
    else:
        return np.exp(- lgamma(Aij + 1))
    

class EM_DCPPM:
    def __init__(self, g, B = None, wr = None, lrs = None, t = None, init_b = None):
        self.g = g
        self.N = g.num_vertices()
        self.E = g.num_edges()
        self.lrs = lrs
        self.A = gt.adjacency(self.g)
        if B is None and init_b is None:
            raise "Error: at least one of B or init_b should be provided!"
        
        if B is not None:
            self.B = B
        else:
            self.B = len(np.unique(init_b))
            
        if init_b is not None:
            self.init_b = init_b
        else:
            self.init_b = [np.random.randint(B) for u in range(self.N)]
            
        if t is not None:
            self.t = t
        else:
            self.t = np.array([B/self.N for u in range(self.N)])
            
        if wr is not None:
            self.wr = wr
        else:
            self.wr = np.ones(self.B)/self.B
    
        self.vm = {}
        perturb = np.random.random()
        degree = self.g.degree_property_map("total")
        for u in self.g.vertices():
            self.vm[u] = np.random.dirichlet((1,1))
            self.vm[u] = np.ones(self.B)/self.B
            if degree[u] == 0:
                continue
            r = np.random.randint(self.B)
            self.vm[u][r] += perturb * 1e-3
            self.vm[u] = self.vm[u]/np.sum(self.vm[u])
            
        self.em = {}
        
        for e in self.g.edges():
            u,v = e
            self.em[(u,v)] = np.ones(self.B)/self.B
            self.em[(v,u)] = np.ones(self.B)/self.B
        
        # prepare the external field
        self.unique_t, self.t_id = np.unique(self.t, return_index = True)
        t2id = {}
        for idx,t in enumerate(self.unique_t):
            t2id[t] = idx
        
        self.t_idx = {}
        for u in self.g.vertices():
            self.t_idx[u] = t2id[self.t[int(u)]]
            
        l = len(self.unique_t)
        self.external_field = np.zeros((l, self.B))        
            
    def bp_iter(self, max_niter = 1, verbose = False, is_randomised = True):
        
#       prepare external filed
        for l in range(self.external_field.shape[0]):
            for r in range(self.B):
                for u in self.g.vertices():
                    temp = 0.0
                    for s in range(self.B):
                        t1 = self.unique_t[l]
                        t2 = self.unique_t[self.t_idx[u]]
                        temp += self.vm[u][s] * h(t1,t2, self.lrs[r,s], 0)
                    self.external_field[l,r] += np.log(temp)
        
        messages_list = []
        for e in self.g.edges():
            u,v = e
            messages_list.append((u,v))
            messages_list.append((v,u))
        if is_randomised:
            random.shuffle(messages_list)
        
        A = gt.adjacency(self.g)
        
        delta = 1.0
        niter = 0
        
        while delta > 1e-3 and niter < max_niter:
            niter += 1
            delta = 0
            for e in messages_list:
                u,v = e        
                
#       get the base message ready
                t_u = self.unique_t[self.t_idx[u]]
                t_v = self.unique_t[self.t_idx[v]]
                base_m_u = np.zeros(self.B)
                for r in range(self.B):
                    temp1 = 0.0
                    temp2 = 0.0
                    for s in range(self.B):
                        temp1 += self.vm[u][s] * h(t_u, t_u, self.lrs[r,s], 0)
                        temp2 += self.vm[v][s] * h(t_v, t_u, self.lrs[r,s], 0 )
                    base_m_u[r] += self.external_field[self.t_idx[u], r] - log(temp1) - log(temp2)
                    
#       update edge-wise message mu_u_v  
                em_temp = np.zeros(self.B)
                for r in range(self.B):
                    for w in self.g.get_all_neighbors(u):
                        if w == v:
                            continue
                        temp1 = 0.0
                        temp2 = 0.0
                        t_u = self.unique_t[self.t_idx[u]]
                        t_w = self.unique_t[self.t_idx[w]]
                        for s in range(self.B):
                            temp1 += self.em[(w,u)][s]*h(t_u, t_w, self.lrs[r,s], A[u,w])
                            temp2 += self.vm[w][s]*h(t_u, t_w, self.lrs[r,s], 0)
                    
                        em_temp[r] += log(temp1) - log(temp2)
                    em_temp[r] += base_m_u[r] + log(self.wr[r])
                
                normalise(em_temp)

                for r in range(self.B):
                    delta += abs(self.em[(u,v)][r] - em_temp[r])
                    self.em[(u,v)][r] = em_temp[r]
                    
#        get the base message ready
                base_m_v = np.zeros(self.B)
                t_v = self.unique_t[self.t_idx[v]]
                for r in range(self.B):
                    base_m_v[r] = 0.0
                    temp = 0.0
                    for s in range(self.B):
                        temp += self.vm[v][s] * h(t_v, t_v, self.lrs[r,s], 0)
                    base_m_v[r] += self.external_field[self.t_idx[v], r] - log(temp)
                        
#        prepare external field to be updated 
                for l in range(self.external_field.shape[0]):
                    for r in range(self.B):
                        temp = 0.0
                        for s in range(self.B):
                            temp += self.vm[v][s] * h(self.unique_t[l], t_v, self.lrs[r,s], 0)
                        self.external_field[l,r] -= log(temp)
                        
#        update node-wise message vm_v
                vm_temp = np.zeros(self.B) 
                for r in range(self.B):
                    for w in self.g.get_all_neighbors(v):
                        if w == v:
                            continue
                        temp1 = 0.0
                        temp2 = 0.0
                        t_w = self.unique_t[self.t_idx[w]]
                        for s in range(self.B):
                            temp1 += self.em[(w,v)][s] * h(t_v, t_w, self.lrs[r,s], A[v,w])
                            temp2 += self.vm[w][s] * h(t_v, t_w, self.lrs[r,s], 0)
                        vm_temp[r] += log(temp1) - log(temp2)
                    vm_temp[r] += log(self.wr[r]) + base_m_v[r]
                
                normalise(vm_temp)
                
                for r in range(self.B):
                    delta += abs(self.vm[v][r] - vm_temp[r])
                    self.vm[v][r] = vm_temp[r]
                
#        update the external field
                for l in range(self.external_field.shape[0]):
                    for r in range(self.B):
                        temp = 0.0
                        for s in range(self.B):
                            temp += self.vm[v][s] * h(self.unique_t[l], t_v, self.lrs[r,s], 0)
                        self.external_field[l,r] += log(temp)
        if verbose:
            print(delta, niter)
            
    def bp_iter_fast(self, max_niter = 1, verbose = False, is_randomised = True):
        
#       prepare external filed
        for l in range(self.external_field.shape[0]):
            for r in range(self.B):
                for u in self.g.vertices():
                    temp = 0.0
                    for s in range(self.B):
                        t1 = self.unique_t[l]
                        t2 = self.unique_t[self.t_idx[u]]
                        temp += self.vm[u][s] * h(t1,t2, self.lrs[r,s], 0)
                    self.external_field[l,r] += np.log(temp)
        
        
#       prepare the connected interaction
        interaction = {}
        for u in self.g.vertices():
            interaction[u] = np.zeros(self.B)
            t_u = self.unique_t[self.t_idx[u]]
            for r in range(self.B):
                for w in self.g.get_all_neighbours(u):
                    if w == u:
                        continue
                    t_w = self.unique_t[self.t_idx[w]]
                    temp_1 = 0.0
                    temp_2 = 0.0
                    
                    for s in range(self.B):
                        temp_1 += self.em[(w,u)][s] * h(t_u, t_w, self.lrs[r,s], self.A[u,w])
                        temp_2 += self.vm[w][s] * h(t_u, t_w, self.lrs[r,s], 0)
                        
                    interaction[u][r] += log(temp_1)
                    interaction[u][r] -= log(temp_2)
                
#       make a copy of the connected interaction
        interaction_messages = interaction.copy()
                    
                    
        messages_list = []
        for e in self.g.edges():
            u,v = e
            messages_list.append((u,v))
            messages_list.append((v,u))
        if is_randomised:
            random.shuffle(messages_list)
        
        delta = 1.0
        niter = 0
        
        while delta > 1e-3 and niter < max_niter:
            niter += 1
            delta = 0
            for e in messages_list:
                u,v = e        
                
#       get the base message ready
                t_u = self.unique_t[self.t_idx[u]]
                t_v = self.unique_t[self.t_idx[v]]
                base_m_u = np.zeros(self.B)
                for r in range(self.B):
                    temp1 = 0.0
                    temp2 = 0.0
                    for s in range(self.B):
                        temp1 += self.vm[u][s] * h(t_u, t_u, self.lrs[r,s], 0)
                        temp2 += self.vm[v][s] * h(t_v, t_u, self.lrs[r,s], 0 )
                    base_m_u[r] += self.external_field[self.t_idx[u], r] - log(temp1) - log(temp2)
                    
#       update edge-wise message mu_u_v  
            
                em_temp = np.zeros(self.B)
                for r in range(self.B):
                    temp1 = 0.0
                    temp2 = 0.0
                    for s in range(self.B):
                        temp1 += self.em[(v, u)][s] * h(t_u, t_v, self.lrs[r,s], self.A[v,u])
                        temp2 += self.vm[v][s] * h(t_u, t_v, self.lrs[r,s], 0)
                    em_temp[r] += interaction[u][r] + base_m_u[r] + log (self.wr[r]) - log(temp1) + log(temp2)
                normalise(em_temp)
                    
#                 em_temp = np.zeros(self.B)
#                 for r in range(self.B):
#                     for w in self.g.get_all_neighbors(u):
#                         if w == v:
#                             continue
#                         temp1 = 0.0
#                         temp2 = 0.0
#                         t_w = self.unique_t[self.t_idx[w]]
#                         for s in range(self.B):
#                             temp1 += self.em[(w,u)][s]*h(t_u, t_w, self.lrs[r,s], self.A[u,w])
#                             temp2 += self.vm[w][s]*h(t_u, t_w, self.lrs[r,s], 0)

#                         em_temp[r] += log(temp1) - log(temp2)
#                     em_temp[r] += base_m_u[r] + log(self.wr[r])

                normalise(em_temp)
                    
#        prepare for updating the interaction term
                for r in range(self.B):
                    temp = 0.0
                    for s in range(self.B):
                        temp += self.em[(u,v)][s] * h(t_u, t_v, self.lrs[r,s], self.A[u,v])
                    interaction_messages[v][r] -= log(temp)

                for r in range(self.B):
                    delta += abs(self.em[(u,v)][r] - em_temp[r])
                    self.em[(u,v)][r] = em_temp[r]

#        update the interaction term
                for r in range(self.B):
                    temp = 0.0
                    for s in range(self.B):
                        temp += self.em[(u,v)][s] * h(t_u, t_v, self.lrs[r,s], self.A[u,v])
                    interaction_messages[v][r] += log(temp)
                    interaction[v][r] = interaction_messages[v][r]
                    
#        get the base message ready
                base_m_v = np.zeros(self.B)
                t_v = self.unique_t[self.t_idx[v]]
                for r in range(self.B):
                    base_m_v[r] = 0.0
                    temp = 0.0
                    for s in range(self.B):
                        temp += self.vm[v][s] * h(t_v, t_v, self.lrs[r,s], 0)
                    base_m_v[r] += self.external_field[self.t_idx[v], r] - log(temp)
                        
#        prepare external field to be updated 
                for l in range(self.external_field.shape[0]):
                    for r in range(self.B):
                        temp = 0.0
                        for s in range(self.B):
                            temp += self.vm[v][s] * h(self.unique_t[l], t_v, self.lrs[r,s], 0)
                        self.external_field[l,r] -= log(temp)

#        prepare updating the interaction term
                for w in self.g.get_all_neighbours(v):
                    t_w = self.unique_t[self.t_idx[w]]
                    for r in range(self.B):
                        temp = 0.0
                        for s in range(self.B):
                            temp += self.vm[v][s] * h(t_v, t_w, self.lrs[r,s], 0)
                        interaction[w][r] += log(temp)
                        
#        update node-wise message vm_v
                vm_temp = np.zeros(self.B) 
                for r in range(self.B): 
                    vm_temp[r] += interaction_messages[v][r] + base_m_v[r] + log(self.wr[r])
                
                normalise(vm_temp)
                
#                 vm_temp = np.zeros(self.B) 
#                 for r in range(self.B):
#                     for w in self.g.get_all_neighbors(v):
#                         if w == v:
#                             continue
#                         temp1 = 0.0
#                         temp2 = 0.0
#                         t_w = self.unique_t[self.t_idx[w]]
#                         for s in range(self.B):
#                             temp1 += self.em[(w,v)][s] * h(t_v, t_w, self.lrs[r,s], self.A[v,w])
#                             temp2 += self.vm[w][s] * h(t_v, t_w, self.lrs[r,s], 0)
#                         vm_temp[r] += log(temp1) - log(temp2)
#                     vm_temp[r] += log(self.wr[r]) + base_m_v[r]
                
#                 print("Old: ", vm_temp)
#                 normalise(vm_temp)
                
                for r in range(self.B):
                    delta += abs(self.vm[v][r] - vm_temp[r])
                    self.vm[v][r] = vm_temp[r]
                
#        update the interaction term
                for w in self.g.get_all_neighbours(v):
                    t_w = self.unique_t[self.t_idx[w]]
                    for r in range(self.B):
                        temp = 0.0
                        for s in range(self.B):
                            temp += self.vm[v][s] * h(t_v, t_w, self.lrs[r,s], 0)
                        interaction[w][r] -= log(temp)
                        interaction_messages[w][r] = interaction[w][r]
                        
#        update the external field
                for l in range(self.external_field.shape[0]):
                    for r in range(self.B):
                        temp = 0.0
                        for s in range(self.B):
                            temp += self.vm[v][s] * h(self.unique_t[l], t_v, self.lrs[r,s], 0)
                        self.external_field[l,r] += log(temp)
        if verbose:
            print(delta, niter)