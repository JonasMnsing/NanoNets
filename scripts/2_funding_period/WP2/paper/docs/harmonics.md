## Time dependent Nanoparticle Networks

### Electrostatics:

We consider a $2$ x $1$ NP network with two electrodes (Source and Drain). The source electrode at voltage $U_S(t)$ is connected to NP1 defined by its potential $\phi_1(t)$ and number of excess charges $Q_1(t)$. NP1 is connected to NP2 defined by $(\phi_2(t),Q_2(t))$. NP2 is connected to the drain electrode at voltage $U_D(t)$. The whole network is put on an insulating enverionment above a substrate which is grounded. Accordingly a NP has a self capacitance relative to ground $C_i = K_s \cdot r_i$ with $K_s$ as a known constant and $r_i$ as the radius of NP $i$. As junctions are insulating, we also have a mutual capacitance between nodes defined as $C_{ij} \approx K_m \frac{r_i r_j}{r_i + r_j + d}$ with $d$ as the shell-to-shell spacing between two nodes and $K_m$ as a known constant. Electrostatics are defined via
$$
\begin{pmatrix}
  Q_1 \\
  Q_2
\end{pmatrix} = 
\bf{C}
\begin{pmatrix}
  \phi_1 \\
  \phi_2
\end{pmatrix} =
\begin{pmatrix}
  C_{11} & -C_{12} \\
  -C_{21} & C_{22}
\end{pmatrix}
\begin{pmatrix}
  \phi_1 \\
  \phi_2
\end{pmatrix}
$$

with **C** as the capacitance matrix and $C_{ii}$ as the sum of capacitance for NP $i$. We get 
$$
C_{11} = C_{S1} + C_{12} + C_{1}
$$
and 
$$
C_{22} = C_{2D} + C_{12} + C_{2}
$$ 
as our two sums for both NPs. Accordingly we get
$$
\vec{\phi}(t) = \bf{C^{-1}}\vec{q}(t) 
$$
for our the potential landscape given the current charge distribution. The inverse of the capacitance matrix is defined as
$$
\bf{C^{-1}}= \frac{1}{\Delta}
\begin{pmatrix}
  C_{22} & C_{12} \\
  C_{21} & C_{11}
\end{pmatrix}
$$
with $\Delta = C_{11}C_{22}-C_{12}^2$.

### Change in free energy

When a charge is tunneling inside the system, we have a change in free energy. The change in free energy for a NP-NP junction is defined as
$$
\Delta F_{i \rightarrow j} = e(\phi_j - \phi_i) + \frac{e^2}{2}(\bf{C_{ii}^{-1}} + \bf{C_{jj}^{-1}} - 2\bf{C_{ij}^{-1}})
$$
and for an Elecrtrode-NP junction
$$
\Delta F_{i \rightarrow E} = e(U_E - \phi_i) + \frac{e^2}{2}\bf{C_{ii}^{-1}}
$$
Both include a work term defining the cost or gain we get of taking one electron against the difference in island potentials. The second part consists of a charging term which defines the change in $\frac{1}{2}Q^{-1}C^{-1}Q$ when an electron is shifted. As an electrode serves as a charge reservoir we lose the penalty in charging energy in the second term and only inlcude the charging energy of the contributing NP. Below all junctions are summarized:
$$
\Delta F_{1 \rightarrow 2} = e(\phi_2 - \phi_1) + \frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta}
$$
$$
\Delta F_{2 \rightarrow 1} = e(\phi_1 - \phi_2) + \frac{e^2}{2}\frac{C_{11}+C_{22}-2C_{12}}{\Delta}
$$
$$
\Delta F_{1 \rightarrow S} = e(U_S - \phi_1) + \frac{e^2}{2}\frac{C_{22}}{\Delta}
$$
$$
\Delta F_{S \rightarrow 1} = e(\phi_1 - U_S) + \frac{e^2}{2}\frac{C_{22}}{\Delta}
$$
$$
\Delta F_{2 \rightarrow D} = e(U_D - \phi_2) + \frac{e^2}{2}\frac{C_{11}}{\Delta}
$$
$$
\Delta F_{D \rightarrow 2} = e(\phi_2 - U_D) + \frac{e^2}{2}\frac{C_{11}}{\Delta}
$$

### Limit of $r_2 \gg r_1$

Assuming the drain attached NP to be much larger than the source attached NP, i.e. $r_2 \gg r_1$ we get $C_{12}\approx K_m \cdot r_1$ and $C_{11}^{-1}\approx \frac{1}{C_{11}}$, $C_{22}^{-1}\approx 0$, $C_{12}^{-1}\approx 0$.
For our set of free energy differences we get:
$$
\Delta F_{1 \rightarrow 2} = e(\phi_2 - \phi_1) + \frac{e^2}{2C_{11}}
$$
$$
\Delta F_{2 \rightarrow 1} = e(\phi_1 - \phi_2) + \frac{e^2}{2C_{11}}
$$
$$
\Delta F_{1 \rightarrow S} = e(U_S - \phi_1) + \frac{e^2}{2C_{11}}
$$
$$
\Delta F_{S \rightarrow 1} = e(\phi_1 - U_S) + \frac{e^2}{2C_{11}}
$$
$$
\Delta F_{2 \rightarrow D} = e(U_D - \phi_2)
$$
$$
\Delta F_{D \rightarrow 2} = e(\phi_2 - U_D)
$$
In this limit the large NP behaves like an equipotential "reservoir" which reduces all charging energy related costs to those of island 1 alone $\frac{e^2}{2C_{11}}$. Effectively our 2-island system reduces to a 1-island system and the large NP becomes clamped by its own self-capacitance having a hardly changing potential upon charging.

### Tunneling Dynamics

For $U_D(t)=0$ and $U_S(t) = U_{0}\cos(\omega_0t)$ we set up the periodic master equation. For every rate we get
$$
\Gamma_{i \rightarrow j}(t)=-\frac{\Delta F_{i \rightarrow j}(t)}{e^2 R_{i \rightarrow j}} \frac{1}{1-\exp(-\Delta F_{i \rightarrow j}(t)/(k_B T))}
$$
When inputting $U_D$ into e.g. \Gamma