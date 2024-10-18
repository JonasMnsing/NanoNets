# Step Input Response
---
## Basics:
For a nanoparticle (NP) network of $N_{NP}$ particles connected to two electrodes, we measure the electric current $I(t)$ response at the *output* electrode upon time varying voltages $U(t)$ at the *input* electrode. The voltage is changed within fixed time steps
$$\Delta t = 10^{-10} \text{ s}$$
while the electric current is evaluated in those steps as the time average across the starting time $t_i$ and the end time $t_j$ of the corresponding step
$$I(t) = \frac{e}{\Delta t} \cdot \sum_{t_0=t_i}^{t_j}(\Gamma_+(t_0) - \Gamma_-(t_0))\cdot t_0$$
with $\Gamma_+$ as the rate for a jump towards the output, $\Gamma_-$ as the rate for a jump from the output, $t_0$ as the time passed for a single jump, and $e$ as the elementary charge. If during the KMC procedure $t_0$ exceeds $t_j$ we reverse the last event, won't consider the correponding rate difference and instead set $t_0 = t_i$ with $t_i$ as the starting time of the next step.

In each simulation we firstly equilibrate the system for $U(t=0)$ to receive the charge landscape $\vec{q}(t=0) := \vec{q}_{eq}$. The equilibrated landscape is stored and $N_{runs} = 500$ parallel simulations for the whole time scale are executed starting with $\vec{q}_{eq}$ as the initial landscape. We will then calculated the mean electric current in each time step $t$ across all simulations
$$\bar{I}(t) = \frac{1}{N_{runs}}\sum_{n=1}^{N_{runs}}I_n(t)$$
and the 95 \% confidence interval for the mean estimate as 
$$\sigma_{\bar{I}}(t) = 1.96 \cdot \frac{\sigma_I}{\sqrt{N_{runs}}}$$
with electric current standard deviation $\sigma_I$.

## Uniform Networks:

<p align='center'>
    a) <img src=../nbk/plots/network.png width="30%"/>
    &nbsp;
    b) <img src=../nbk/plots/I_O.png width="63%"/>
    a) Network with 49 NPs and two electrodes marked red. b) Output electric current response for variable network sizes. The input electrode voltage signal is marked blue. 
</p>

Firstly we analyse the output response in a uniform network of nanoparticles with equal nanoparticle sizes (electrostatic properties) and resistances (tunneling properties). The input voltage is switched based on $U \in \{100 \text{ mV}, 200 \text{ mV}\}$. The upper plots shows an example network and the responses for variable network sizes. We detect an exponential decline after the input voltage was reduced. When subtracting the y-axis offset of this decline, we are able to fit
$$I(t) = I_0 \cdot e^{\frac{t}{\tau}} \Leftrightarrow \ln(I(t)) = ln(I_0) + \frac{t}{\tau}$$
and achieve the time relaxation time displayed above. We can use the same fitting process for the nanoparticle potentials:

<div align='center'>
    <img src=../nbk/plots/node_signal_time_scales.png width="50%"/>
    <p> For a network of 49 NPs the distribution indicates variable relaxation times across the nanoparticles. Output relaxation time marked black. </p>
</div>

### Disordered Networks

We repeat the upper analysis for networks with disordered properties. There are three properties which can be coosen to be disordered:

| Topology  | Resistances | Nanoparticle Sizes  |
| --------  | ----------- | ------------------  |
| Network of $N_{NP} = 49$ NPs connected based on a *random regular graph* with node degree $N_j = 4$. | For a cubic shaped network of $N_{NP} = 49$ Nps all junction resistances are initially set at $25$ M$\Omega$. Next $9$ NPs are choosen at random based on a given *seed*. Junctions connected to those NPs get a different resistance of $R$. | For a cubic shaped network of $N_{NP} = 49$ Nps all nanoparticle sizes are set at $10$ nm. Next $9$ NPs are choosen at random based on a given *seed*. Those NPs get a different radius of $r$.

<p align='center'>
    <img src=../nbk/plots/relaxation_disorder.png width="100%"/>
    Relaxation time for different distributions of disorder in terms of nanoparticle sizes and resistances. Black dashed line indicates uniform network. 
</p>

<!-- <p align='center'>
    a) <img src=../nbk/plots/I_O_R.png width="95%"/>
    b) <img src=../nbk/plots/I_O_radius.png width="95%"/>
    Output electric current responses for disorder in resistance (a) or radius (b). Each simulation was repeated 10 times with varying distributions of high resistive junctions or large NPs. 
</p> -->