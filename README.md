# belief_propagation_dcsbm
This repository provides an implementation of the belief propagtion (BP) algorithm for fitting the degree-corrected stochastic block model (DC-SBM) defined in [1]. Such implementation allows practitioners to conduct ***community detection*** in networks with heterogeneous degree distribution. This implementation is built on the <a href="https://graph-tool.skewed.de/">graph-tool library</a>[2]. The main motivation behind this repository is to demonstrate how to put the BP algorithm into practice. We hope this could be helpful for those who come across the BP for the first time and like to build their own implementations.

Run the following code to see whether the BP algorithm is really working.
```
python3.9 text_bp_accuracy.py

# output

Is modified: False; Partition overlap before BP: 0.51
Is modified: False; Partition overlap after BP: 0.945
```

Run the following code to see difference in running time between different ways of updating the BP equations. This comparison is related to two different ways of updating the BP equations as to be explained below. `gen_dcppm`
```
python3.9 text_compare_bp_running_time.py

# output

One iteration of BP with the original update scheme takes: 11.91773271560669 secs
One iteration of BP with the improved update scheme takes: 5.926676273345947 secs
```

### Why BP
Belief propagation allows direct computation of the marginal probability distribution of variables in graphs. For networks generated from the stochastic block models(SBMs), assigning nodes into group according to the marginal probability distribution is ***optimal*** in terms of the number of nodes bring corrected labeled. Although it is possible to obtain the same marginal by drawing samples via MCMC, belief propagation is more efficient with a computation complexity which is linearly dependant on the size of the system.

Computing the marginal probability with BP requires the parameters of the model as input, which are usually not available when it comes to empirical networks. Nevetheless, we can take the ***expectaction-maximisation (EM)*** scheme and use the BP algroithm for the expectaction step[3].

Another important application of the BP algorithm is in studying the phase-transition phenomenon in of high-dimensional inference problems. In particular, for community detection with SBMs, BP plays an important role in advancing our understanding about the detectability of planted community structure[3-4].

### What is going on here
The main pupose of this repository is to provide a reference for anyone who like to write his/her first BP implementation. Although the BP algorithm is elegant and easy to comprehend, for people who are new to the algorithm, the learning curve could be steep for implmenting the algorithm. There are several existed packages for implementing BP (see the list below), but none of them are done in Python. This is mainly due to the efficiency consideration. As a result, existed resources are not straight-forward for someone who only has background in Python but like to understand BP and its implementation. We hope the examples provided here could be a helpful reference for picking up the key steps in the implementation of BP for community dectection. 

Moreover, we like to add a commment on the computation complexity of BP. There is a nuance between applying BP to networks with homogenous and heterogeneous degree distribution. The nuance is related to the fact that there are many unnecessarily repeated computations could have been avoided when we update BP equations. When the averaged degree is fixed, the amount of repeated computations is more problematic in network with heterogeneous degree distribution than the homogeneous case. We can save almost a half of the computation time if we adopt a modfied way of updating the BP equations, as shown in the following figure. Up to the time when this repository is produced, ***none*** of any existed BP implementations has included taken this nucance into account. 

<p align="center">
<img src="bp_running_time_comparison.png" width=450><br>
<b>Fig.1: Comparison of the BP running time with different update scheme.</b>
</p>

We demonstrate the difference in running time between the original and the improved update scheme of BP in Fig.1[^1]. To generate networks with heterogeneous degree distribution, we set the degree propensity value <img src="https://latex.codecogs.com/svg.image?\inline&space;\{\theta_u\}" title="\inline \{\theta_u\}" /> as follows,

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\theta_{u&plus;nN/B}&space;=&space;\theta_{u}&space;=&space;\frac{x_{u}}{\sum_{v}^{N/B}&space;x_{v}},&space;\;\;&space;\forall&space;u&space;\in&space;\{1,2,..,N/B\},&space;\;\;&space;\forall&space;n&space;\in&space;\{1,2,..,B-1\}." title="\theta_{u+nN/B} = \theta_{u} = \frac{x_{u}}{\sum_{v}^{N/B} x_{v}}, \;\; \forall u \in \{1,2,..,N/B\}, \;\; \forall n \in \{1,2,..,B-1\}." />
</p>

where <img src="https://latex.codecogs.com/svg.image?\inline&space;\{x_u\}" title="\inline \{x_u\}" /> are samples drawn from the truncated Zipf's law

<p align="center">
<img src="https://latex.codecogs.com/svg.image?f_{X}(x)&space;=&space;\begin{cases}x^{-\zeta}/\sum^{x_{\text{max}}}_{x&space;=&space;1}x^{-\zeta},&space;\;\;&space;\text{if&space;}&space;1&space;\leq&space;x&space;\leq&space;x_{\text{max}};&space;x\in&space;\mathcal{Z}&space;\\0,&space;\;\;&space;\text{o.w.}\end{cases}." title="f_{X}(x) = \begin{cases}x^{-\zeta}/\sum^{x_{\text{max}}}_{x = 1}x^{-\zeta}, \;\; \text{if } 1 \leq x \leq x_{\text{max}}; x\in \mathcal{Z} \\0, \;\; \text{o.w.}\end{cases}." />
</p>
The comparison is done in networks with N = 10^5 nodes, B = 2 communities, and average degree equal to 5. We chose the cut-off value 50 for the truncated Zipfs' distribution. 

[^1]:Results are obtained with a C++ implementation which is more efficient than the examples provided in this repository. However, making Python and C++ work together requires more work on the setup, so we only present the results here. 


<br><br>
### The unncessarily repeated computations in updating BP
The BP algorithm for the DC-SBM requires to update a series of BP equations,

<p align="center">
<img src="https://latex.codecogs.com/svg.image?&space;\mu_r^{u&space;\rightarrow&space;v}&space;=&space;\frac{\gamma_r}{Z^{u&space;\rightarrow&space;v}}&space;e^{-H_{r}}&space;\prod_{w&space;\in&space;\partial&space;u&space;\setminus&space;v}&space;\frac{\sum_{s=1}^B&space;\mu_s^{w&space;\rightarrow&space;u}&space;g(\theta_w,&space;\theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B&space;\mu_s^w&space;g(\theta_w,&space;\theta_u,\lambda_{rs},0)}," title=" \mu_r^{u \rightarrow v} = \frac{\gamma_r}{Z^{u \rightarrow v}} e^{-H_{r}} \prod_{w \in \partial u \setminus v} \frac{\sum_{s=1}^B \mu_s^{w \rightarrow u} g(\theta_w, \theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B \mu_s^w g(\theta_w, \theta_u,\lambda_{rs},0)}," />
</p>

where the function *g* is the Poisson probability 
<p align="center">
<img src="https://latex.codecogs.com/svg.image?g(\theta_{u},&space;\theta_{v},&space;\lambda_{rs},&space;A_{uv})&space;=&space;e^{-\theta_{u}\theta_{v}\lambda_{rs}}&space;\frac{(\theta_{u}\theta_{v}\lambda_{rs})^{A_{uv}}}{A_{uv}!}" title="g(\theta_{u}, \theta_{v}, \lambda_{rs}, A_{uv}) = e^{-\theta_{u}\theta_{v}\lambda_{rs}} \frac{(\theta_{u}\theta_{v}\lambda_{rs})^{A_{uv}}}{A_{uv}!} ," />

and the <img src="https://latex.codecogs.com/svg.image?H_r" title="H_r," /> is defined as 

<p align="center">
<img src="https://latex.codecogs.com/svg.image?&space;H_{r}&space;=&space;-&space;\sum_w&space;\log&space;\large(\sum_{s&space;=1}^{B}&space;\mu_s^w&space;g(\theta_w,\theta_u,\lambda_{rs},0)\large)" title=" H_{r} = - \sum_w \log \large(\sum_{s =1}^{B} \mu_s^w g(\theta_w,\theta_u,\lambda_{rs},0)\large) ." />
</p>

The marginal probability distribution of the node *u* is given by 

<p align="center">
<img src="https://latex.codecogs.com/svg.image?&space;\mu_r^u&space;=&space;\frac{\gamma_r}{Z^u}&space;e^{-H_{r}}&space;\prod_{w&space;\in&space;\partial&space;u}&space;\frac{\sum_{s=1}^B&space;\mu_s^{w&space;\rightarrow&space;u}&space;g(\theta_w,&space;\theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B&space;\mu_s^w&space;g(\theta_w,&space;\theta_u,\lambda_{rs},0)}." title=" \mu_r^u = \frac{\gamma_r}{Z^u} e^{-H_{r}} \prod_{w \in \partial u} \frac{\sum_{s=1}^B \mu_s^{w \rightarrow u} g(\theta_w, \theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B \mu_s^w g(\theta_w, \theta_u,\lambda_{rs},0)}." />
</p>

### Other implementation of BP for community dection
The following packages/softwares should provide better performance in terms of efficiency:

- <a href="https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.inference.EMBlockState/">graph-tool</a>: a python library with algorihtms being implemented in C++ by <a href="https://skewed.de/tiago">Tiago Peixto</a>

- <a href="https://github.com/everyxs/SBMbp/releases">Bayesian model selection of Stochastic Block Model</a>: a java implementation by <a href="https://xiaoranyan.wordpress.com/">Xiaoran Yan</a>

- Modnet: a C++ implementation by <a href="http://home.itp.ac.cn/~panzhang/">Pang Zhang</a>

- <a href="https://github.com/junipertcy/sbm-bp">sbm-bp</a>: a C++ implementation by <a href="https://junipertcy.info/">Tzu-Chi Yen</a> 

<br><br>
#### References:
<p><a>[1] X. Yan, J. E. Jensen, F. Krzakala, C. Moore, C. R. Shalizi,
L. Zdeborova, P. Zhang, and Y. Zhu, Model Selection for
Degree-Corrected Block Models, 2014.</a>
<p><a>[2] Tiago P. Peixoto. The graph-tool python library. figshare, 2014. </a>
<p><a>[3] Decelle, Aurelien, et al. "Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications." Physical Review E 84.6 (2011): 066106.</a>
<p><a>[4] Zhang, Pan, Cristopher Moore, and M. E. J. Newman. "Community detection in networks with unequal groups." Physical review E 93.1 (2016): 012303.
