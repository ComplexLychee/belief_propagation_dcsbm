from functions import check_bp_accuracy

nr = 100
B = 2 
k = 5
ep = 0.8
gamma = 2.7
xmax = 1

check_bp_accuracy(nr, B, k, ep, gamma, xmax, is_modified = False, max_niter = 1e3)
