# belief_propagation_dcsbm
This repository provides an implementation of the belief propagtion (BP) algorithm for fitting the degree-corrected stochastic block model (DC-SBM) as defined in [1]. This implementation is built on the <a href="https://graph-tool.skewed.de/">graph-tool library</a>[2]. The main motivation behind this repository is to demonstrate how to put the BP algorithm into practice. We hope this could be helpful for those who come across the BP for the first time and like to build their own implementations.

Run the following code to see whether the BP algorithm is really working.
```
python3.9 text_bp_accuracy.py

# output

Is modified: False Partition overlap before BP: 0.51
Is modified: False Partition overlap after BP: 0.945
```

Run the following code to see difference in running time between different ways of updating the BP equations. This comparison is related to two different ways of updating the BP equations as to be explained below.
```
python3.9 text_compare_bp_running_time.py

# output

One iteration of BP with the original update scheme takes: 11.91773271560669 secs
One iteration of BP with the improved update scheme takes: 5.926676273345947 secs
```

<p align="center">
<img src="bp_running_time_comparison.png" width=450><br>
<b>Comparison of the BP running time with different update scheme. Results are obtained with a C++ implementation which is more efficient than the one available in this repository. However, making Python and C++ work together requires more careful setup, so we only present the results here.  </b>
</p>
<br/><br/>

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

#### References in the table:
<p><a>[1] X. Yan, J. E. Jensen, F. Krzakala, C. Moore, C. R. Shalizi,
L. Zdeborova, P. Zhang, and Y. Zhu, Model Selection for
Degree-Corrected Block Models, 2014.</a>
<p><a>[2] Tiago P. Peixoto. The graph-tool python library. figshare, 2014. </a>
