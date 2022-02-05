# belief_propagation_dcsbm
This repository is produced to serve as supplemental materials for the thesis ***Realistic constraints, model selection, and detectability of modular network structures***.

Here we provide an implementation of the belief propagtion (BP) algorithm for fitting the degree-corrected stochastic block model (DC-SBM) defined in [1]. Such implementation allows practitioners to conduct ***community detection*** in networks with heterogeneous degree distribution. This implementation is built on the <a href="https://graph-tool.skewed.de/">graph-tool library</a>[2]. The main motivation behind this repository is to demonstrate how to put the BP algorithm into practice. We hope this could be helpful for those who come across the BP for the first time and like to build their own implementations.

Run the following code to see whether the BP algorithm is really working.
```
python3.9 test_bp_accuracy.py

# output

Is modified: False; Partition overlap before BP: 0.51
Is modified: False; Partition overlap after BP: 0.945
```

Run the following code to see difference in running time between different ways of updating the BP equations. This comparison is related to two different ways of updating the BP equations as to be explained below.
```
python3.9 test_compare_bp_running_time.py

# output

One iteration of BP with the original update scheme takes: 11.91773271560669 secs
One iteration of BP with the improved update scheme takes: 5.926676273345947 secs
```

### Why BP
Belief propagation allows direct computation of the marginal probability distribution of variables in graphs. For networks generated from the stochastic block models(SBMs), assigning nodes into group according to the marginal probability distribution is ***optimal*** in terms of the number of nodes bring corrected labeled. Although it is possible to obtain the same marginal by drawing samples via MCMC, belief propagation is more efficient with a computation complexity which is linearly dependant on the size of the system.

Computing the marginal probability with BP requires the parameters of the model as input, which are usually not available when it comes to empirical networks. Nevetheless, we can take the ***expectaction-maximisation (EM)*** scheme and use the BP algroithm for the expectaction step[3].

Another important application of the BP algorithm is in studying the phase-transition phenomenon in of high-dimensional inference problems. In particular, for community detection with SBMs, BP plays an important role in advancing our understanding about the detectability of planted community structure[3-4].

### What is going on here
The main pupose of this repository is to provide a reference for anyone who wants to write his/her first BP implementation. Although the BP algorithm is elegant and easy to comprehend, for people who are new to the algorithm, the learning curve could be steep for implmenting the algorithm. There are several existed packages for implementing BP (see a list below), but none of them are done in Python. This is mainly due to the efficiency consideration. As a result, existed resources are not straight-forward for someone who only has background in Python but likes to understand BP and its implementation. We hope the examples provided here could be a helpful reference for picking up the key steps in the implementation of BP for community dectection. 

Moreover, we like to add a commment on the computation complexity of BP. There is a nuance between applying BP to networks with homogenous and heterogeneous degree distribution. The nuance is related to the fact that there are many unnecessarily repeated computations could have been avoided when we update BP equations. When the averaged degree is fixed, the amount of repeated computations is more problematic in network with heterogeneous degree distribution than the homogeneous case. We can save almost a half of the computation time if we adopt a modfied way of updating the BP equations, as shown in the following figure. Up to the time when this repository is produced, ***none*** of any existed BP implementations has included taken this nucance into account. 

<p align="center">
<img src="/pics/bp_running_time_comparison.png" width=450><br>
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
Our comparison is done in networks with N = 10^5 nodes, B = 2 communities, and average degree equal to 5. We chose the cut-off value 50 for the truncated Zipfs' distribution. As shown in Fig.1, when the average degree is fixed, the adavantgae of the improved updating scheme becomes more clear as as the level of heterogeneity of degree distribution increases. Even when the degree propensity parameters are set to be uniform (when the shape parameter is set to infinity),the improved scheme requires less running time than the original scheme. For details about the difference between the two update scheme, see next section below.

[^1]:Results are obtained with a C++ implementation which is more efficient than the examples provided in this repository. However, making Python and C++ work together requires more work on the setup, so we only present the results here. 


<br><br>
### The unncessarily repeated computations in updating BP
To explain the reapeated computations which have been overlooked by existed implementation of BP, we firstly recall the BP messages to be updated for the DC-SBM: 

<p align="center">
 <img src="https://latex.codecogs.com/svg.image?\mu_r^{u&space;\rightarrow&space;v}&space;=&space;\frac{\gamma_r}{Z^{u&space;\rightarrow&space;v}}&space;e^{-H_{r}}&space;\prod_{w&space;\in&space;\partial&space;u&space;\setminus&space;v}&space;\frac{\sum_{s=1}^B&space;\mu_s^{w&space;\rightarrow&space;u}&space;g(\theta_w,&space;\theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B&space;\mu_s^w&space;g(\theta_w,&space;\theta_u,\lambda_{rs},0)},&space;\;\;&space;(1)" title="\mu_r^{u \rightarrow v} = \frac{\gamma_r}{Z^{u \rightarrow v}} e^{-H_{r}} \prod_{w \in \partial u \setminus v} \frac{\sum_{s=1}^B \mu_s^{w \rightarrow u} g(\theta_w, \theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B \mu_s^w g(\theta_w, \theta_u,\lambda_{rs},0)}, \;\; (1)" /> 
 </p>
 
where <img src="https://latex.codecogs.com/svg.image?\inline&space;\partial&space;u" title="\inline \partial u" /> is the neighbouring set of node <img src="https://latex.codecogs.com/svg.image?\inline&space;u" title="\inline u" />, and the function *g* is the Poisson probability 
<p align="center">
<img src="https://latex.codecogs.com/svg.image?g(\theta_{u},&space;\theta_{v},&space;\lambda_{rs},&space;A_{uv})&space;=&space;e^{-\theta_{u}\theta_{v}\lambda_{rs}}&space;\frac{(\theta_{u}\theta_{v}\lambda_{rs})^{A_{uv}}}{A_{uv}!}.&space;\;\;&space;(2)" title="g(\theta_{u}, \theta_{v}, \lambda_{rs}, A_{uv}) = e^{-\theta_{u}\theta_{v}\lambda_{rs}} \frac{(\theta_{u}\theta_{v}\lambda_{rs})^{A_{uv}}}{A_{uv}!}. \;\; (2)" />
</p>

The <img src="https://latex.codecogs.com/svg.image?H_r" title="H_r," /> is defined as 

<p align="center">
<img src="https://latex.codecogs.com/svg.image?&space;H_{r}&space;=&space;-&space;\sum_w&space;\log&space;\large(\sum_{s&space;=1}^{B}&space;\mu_s^w&space;g(\theta_w,\theta_u,\lambda_{rs},0)\large)&space;\;\;&space;(3)" title=" H_{r} = - \sum_w \log \large(\sum_{s =1}^{B} \mu_s^w g(\theta_w,\theta_u,\lambda_{rs},0)\large) \;\; (3)" />
</p>
For detailed derivation please refer to the paper by Xiaoran Yan[1].
<br>
Now consider a node *u* in the network and the messages sending out to its two neighbours, say _v1_ and _v2_. Many terms are repeatedly computeted in the product in the equation (1). Specifically, the following ratio value is the same but will be recomputed when we update messages sending from _u_ to _v1_ and _v2_,  

<p align="center">
 <img src="https://latex.codecogs.com/svg.image?\frac{\sum_{s=1}^B&space;\mu_s^{w&space;\rightarrow&space;u}&space;g(\theta_w,&space;\theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B&space;\mu_s^w&space;g(\theta_w,&space;\theta_u,\lambda_{rs},0)},&space;\;\;&space;(4)" title="\mu_r^{u \rightarrow v} = \frac{\gamma_r}{Z^{u \rightarrow v}} e^{-H_{r}} \prod_{w \in \partial u \setminus v} \frac{\sum_{s=1}^B \mu_s^{w \rightarrow u} g(\theta_w, \theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B \mu_s^w g(\theta_w, \theta_u,\lambda_{rs},0)}, \;\; (4)" /> 
</p>

for any node *w* which is also a neightour of node *u*.
<p align="center">
<img src="/pics/bp_repeated_computations.png" width=300><br>
<b>Fig.2: Nodes on the left-hand-side of grey dashed arc are neighbours of node _u_ except for the two nodes _v1_ and _v2_. They send the same information to the node _u_ when we update the two messages sending out from _u_ to _v1_ and _v2_. </b>
</p>

The amount of wasted computations is roughly at the scale of 

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\sum_{(u,v)&space;\in&space;\mathcal{E}}&space;k_{u}B&space;=&space;\sum_{u}^{N}&space;k_{u}^{2}B.&space;\;\;&space;(5)" title="\sum_{(u,v) \in \mathcal{E}} k_{u}B = \sum_{u}^{N} k_{u}^{2}B. \;\; (5)" />
</p>
   
which could be prohibitively expensive, especially in networks with large-degree of nodes. This issue can be addressed by precomputing and maintaining the interactions between every node and all its neighbours,

<p align="center">
<img src="https://latex.codecogs.com/svg.image?\mathcal{I}^{u}_{r}&space;=&space;\prod_{w&space;\in&space;\partial&space;u}\frac{\sum_{s=1}^B&space;\mu_s^{w&space;\rightarrow&space;u}&space;g(\theta_w,&space;\theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B&space;\mu_s^w&space;g(\theta_w,&space;\theta_u,\lambda_{rs},0)}.\;\;(6)" title="\mathcal{I}^{u}_{r} = \prod_{w \in \partial u}\frac{\sum_{s=1}^B \mu_s^{w \rightarrow u} g(\theta_w, \theta_u,\lambda_{rs},A_{uw})}{\sum_{s=1}^B \mu_s^w g(\theta_w, \theta_u,\lambda_{rs},0)}.\;\;(6)" />
</p>

We have implmented both of the original and the improved ways for updating BP equations. The different in running time between the two update scheme can be seen by excuting the `test_compare_bp_running_time.py` file. One can play around with the parameters of the simulation to see how does the difference between the two changes. For example, as shown in Fig.1, if one change shape parameter of the Zipf's distribution <img src="https://latex.codecogs.com/svg.image?\zeta" title="\zeta" />, the degree distribution should becomes more heterogeneous, making the advantage of the improved updating scheme more clear. 

### Other implementation of BP for community dection
If you have your data and like to apply BP to analyse your data, you might want check one of the following available packages:

- <a href="https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.inference.EMBlockState/">graph-tool</a>: a python library with algorihtms being implemented in C++

- <a href="https://github.com/everyxs/SBMbp/releases">Bayesian model selection of Stochastic Block Model</a>: a java implementation
 
- Modnet: a C++ implementation

- <a href="https://github.com/junipertcy/sbm-bp">sbm-bp</a>: a C++ implementation

<br><br>
#### References:
<p><a>[1] X. Yan, J. E. Jensen, F. Krzakala, C. Moore, C. R. Shalizi,
L. Zdeborova, P. Zhang, and Y. Zhu, Model Selection for
Degree-Corrected Block Models, 2014.</a>
<p><a>[2] Tiago P. Peixoto. The graph-tool python library. figshare, 2014. </a>
<p><a>[3] Decelle, Aurelien, et al. "Asymptotic analysis of the stochastic block model for modular networks and its algorithmic applications." Physical Review E 84.6 (2011): 066106.</a>
<p><a>[4] Zhang, Pan, Cristopher Moore, and M. E. J. Newman. "Community detection in networks with unequal groups." Physical review E 93.1 (2016): 012303.
